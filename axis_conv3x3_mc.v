//=============================================================================
// 多通道 3x3 卷积加速器 (axis_conv3x3_mc)
// 支持参数化输入/输出通道数，定点量化（右移+饱和截断）
// AXI-Stream 接口，单时钟域
//=============================================================================
module axis_conv3x3_mc #(
    parameter IN_CH      = 1,    // 输入通道数
    parameter OUT_CH     = 6,    // 输出通道数
    parameter DATA_W     = 8,    // 输入/输出数据位宽（有符号）
    parameter WEIGHT_W   = 8,    // 权重位宽（有符号）
    parameter BIAS_W     = 16,   // 偏置位宽（有符号）
    parameter ACC_W      = 32,   // 累加器位宽
    parameter OUT_W      = 8,    // 输出数据位宽
    parameter SHIFT_BITS = 4,    // 累加器算术右移位数
    parameter IMG_W      = 8,    // 输入图像宽度
    parameter IMG_H      = 8     // 输入图像高度
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //==================== 输入 AXI-Stream ====================
    input  wire signed [DATA_W-1:0]    s_axis_tdata,
    input  wire                          s_axis_tvalid,
    output reg                           s_axis_tready,
    input  wire                          s_axis_tlast,

    //==================== 输出 AXI-Stream ====================
    output reg signed [OUT_W-1:0]    m_axis_tdata,
    output reg                         m_axis_tvalid,
    input  wire                         m_axis_tready,
    output reg                         m_axis_tlast,

    //==================== 权重配置接口 ====================
    input  wire                          weight_wr_en,
    input  wire [2:0]                    weight_out_ch,   // 输出通道索引 0~OUT_CH-1
    input  wire [3:0]                    weight_addr,     // 权重地址 0~8 (3x3)
    input  wire signed [WEIGHT_W-1:0]   weight_data,

    //==================== 偏置配置接口 ====================
    input  wire                          bias_wr_en,
    input  wire [2:0]                    bias_out_ch,     // 输出通道索引 0~OUT_CH-1
    input  wire signed [BIAS_W-1:0]     bias_data
);

//=============================================================================
// 本地参数
//=============================================================================
localparam OUT_IMG_W = IMG_W - 2;   // 输出宽度 (无 padding, stride=1)
localparam OUT_IMG_H = IMG_H - 2;   // 输出高度
localparam COL_W     = $clog2(IMG_W);   // 列计数器位宽
localparam ROW_W     = $clog2(IMG_H);   // 行计数器位宽
localparam CH_W      = $clog2(OUT_CH);  // 通道计数器位宽
localparam OUT_PIXEL_CNT = OUT_CH * OUT_IMG_W * OUT_IMG_H; // 总输出像素数

//=============================================================================
// 内部信号
//=============================================================================
reg [COL_W-1:0]  col_cnt;          // 当前输入列计数器
reg [ROW_W-1:0]  row_cnt;          // 当前输入行计数器
reg [CH_W-1:0]   out_ch_cnt;       // 输出通道计数器
reg [15:0]        out_pixel_cnt;    // 输出像素总计数器（用于 tlast）
reg                 frame_done;       // 一帧输入完成标志，等最后一个窗口输出完再回 IDLE

// 状态机
localparam IDLE     = 2'd0;
localparam INPUT    = 2'd1;
localparam OUTPUT   = 2'd2;
reg [1:0] state;

wire window_valid;    // 3x3 窗口有效
wire line_end;        // 一行结束
wire frame_end;       // 一帧结束

//=============================================================================
// Line Buffer: 3 个行 bank 轮转，存储前 2 行 + 当前行
//=============================================================================
reg signed [DATA_W-1:0] line_buf [0:2][0:IMG_W-1];  // 3 行 bank
reg [1:0] wr_bank;    // 当前写入 bank (0,1,2 轮转)

wire [1:0] prev1_bank = (wr_bank + 2'd2) % 3'd3;  // 前一行 bank
wire [1:0] prev2_bank = (wr_bank + 2'd1) % 3'd3;  // 前前行 bank

//=============================================================================
// 当前行移位寄存器：存储最近 3 个输入像素，用于窗口提取
//=============================================================================
reg signed [DATA_W-1:0] cur_line [0:2];  // cur_line[0]=最新, cur_line[2]=最旧

//=============================================================================
// 3x3 窗口提取
//=============================================================================
wire signed [DATA_W-1:0] win [0:2][0:2];  // win[row][col], row0=最旧行

// 前前行（3 列）
assign win[0][0] = line_buf[prev2_bank][col_cnt - 2'd2];
assign win[0][1] = line_buf[prev2_bank][col_cnt - 2'd1];
assign win[0][2] = line_buf[prev2_bank][col_cnt];

// 前一行（3 列）
assign win[1][0] = line_buf[prev1_bank][col_cnt - 2'd2];
assign win[1][1] = line_buf[prev1_bank][col_cnt - 2'd1];
assign win[1][2] = line_buf[prev1_bank][col_cnt];

// 当前行（3 列）
// 注意：cur_line 在当前周期被非阻塞赋值更新，posedge 采样时读到的是旧值
//   旧 cur_line[0] = col-1, 旧 cur_line[1] = col-2, 旧 cur_line[2] = col-3
// 所以正确映射：win[2][0]=旧cur_line[1]=col-2, win[2][1]=旧cur_line[0]=col-1, win[2][2]=s_axis_tdata=col
assign win[2][0] = cur_line[1];   // col-2
assign win[2][1] = cur_line[0];   // col-1
assign win[2][2] = s_axis_tdata;  // col（当前输入像素）

// 窗口有效：行 >=2 且 列 >=2
assign window_valid = (row_cnt >= 2'd2) && (col_cnt >= 2'd2);
assign line_end  = (col_cnt == IMG_W - 1);
assign frame_end = (row_cnt == IMG_H - 1) && line_end;

//=============================================================================
// 权重存储器: OUT_CH x 9 x WEIGHT_W
//=============================================================================
reg signed [WEIGHT_W-1:0] weight_mem [0:OUT_CH-1][0:8];

// 权重写入
always @(posedge clk) begin
    if (weight_wr_en) begin
        weight_mem[weight_out_ch][weight_addr] <= weight_data;
    end
end

//=============================================================================
// 偏置存储器: OUT_CH x BIAS_W
//=============================================================================
reg signed [BIAS_W-1:0] bias_mem [0:OUT_CH-1];

always @(posedge clk) begin
    if (bias_wr_en) begin
        bias_mem[bias_out_ch] <= bias_data;
    end
end

//=============================================================================
// PE 阵列：每个输出通道一个 PE（9 乘法 + 累加 + 偏置 + 移位 + 饱和）
//=============================================================================
wire signed [ACC_W-1:0] pe_acc [0:OUT_CH-1];    // 累加结果
wire signed [OUT_W-1:0] pe_result [0:OUT_CH-1];  // 最终饱和结果

genvar oc;
generate
    for (oc = 0; oc < OUT_CH; oc = oc + 1) begin : gen_pe
        // 9 个乘法结果
        wire signed [DATA_W+WEIGHT_W-1:0] prod [0:8];
        assign prod[0] = win[0][0] * weight_mem[oc][0];
        assign prod[1] = win[0][1] * weight_mem[oc][1];
        assign prod[2] = win[0][2] * weight_mem[oc][2];
        assign prod[3] = win[1][0] * weight_mem[oc][3];
        assign prod[4] = win[1][1] * weight_mem[oc][4];
        assign prod[5] = win[1][2] * weight_mem[oc][5];
        assign prod[6] = win[2][0] * weight_mem[oc][6];
        assign prod[7] = win[2][1] * weight_mem[oc][7];
        assign prod[8] = win[2][2] * weight_mem[oc][8];

        // 累加树：9 个乘积 + 偏置
        assign pe_acc[oc] = bias_mem[oc]
                          + prod[0] + prod[1] + prod[2]
                          + prod[3] + prod[4] + prod[5]
                          + prod[6] + prod[7] + prod[8];

        // 算术右移 + 饱和截断
        conv_saturate #(
            .IN_W      (ACC_W),
            .OUT_W     (OUT_W),
            .SHIFT_BITS(SHIFT_BITS)
        ) u_sat (
            .data_in  (pe_acc[oc]),
            .data_out (pe_result[oc])
        );
    end
endgenerate

//=============================================================================
// 输出寄存器阵列：窗口有效时锁存 6 通道结果
//=============================================================================
reg signed [OUT_W-1:0] out_reg [0:OUT_CH-1];
reg                     out_reg_valid;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        out_reg_valid <= 1'b0;
    end else begin
        if (window_valid && s_axis_tvalid && s_axis_tready) begin
            out_reg_valid <= 1'b1;
        end else if (state == OUTPUT && out_ch_cnt == OUT_CH-1 && m_axis_tready && m_axis_tvalid) begin
            out_reg_valid <= 1'b0;
        end
    end
end

genvar oc2;
generate
    for (oc2 = 0; oc2 < OUT_CH; oc2 = oc2 + 1) begin : gen_out_reg
        always @(posedge clk) begin
            if (window_valid && s_axis_tvalid && s_axis_tready) begin
                out_reg[oc2] <= pe_result[oc2];
            end
        end
    end
endgenerate

//=============================================================================
// 主状态机
//=============================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state         <= INPUT;
        col_cnt       <= 'd0;
        row_cnt       <= 'd0;
        out_ch_cnt    <= 'd0;
        out_pixel_cnt <= 'd0;
        frame_done    <= 1'b0;
        s_axis_tready <= 1'b1;
        m_axis_tvalid <= 1'b0;
        m_axis_tdata  <= 'd0;
        m_axis_tlast  <= 1'b0;
        wr_bank       <= 2'd0;
    end else begin
        case (state)
            //--------------------------------------------------------
            IDLE: begin
                s_axis_tready <= 1'b1;
                m_axis_tvalid <= 1'b0;
                m_axis_tlast  <= 1'b0;
                if (s_axis_tvalid) begin
                    state <= INPUT;
                end
            end

            //--------------------------------------------------------
            INPUT: begin
                if (s_axis_tvalid && s_axis_tready) begin
                    // 写入 Line Buffer 当前 bank
                    line_buf[wr_bank][col_cnt] <= s_axis_tdata;

                    // 更新当前行移位寄存器
                    cur_line[2] <= cur_line[1];
                    cur_line[1] <= cur_line[0];
                    cur_line[0] <= s_axis_tdata;

                    // 列计数
                    if (line_end) begin
                        // 新的一行开始前清空当前行移位寄存器，避免混入上一行行尾数据
                        cur_line[0] <= 'd0;
                        cur_line[1] <= 'd0;
                        cur_line[2] <= 'd0;
                        col_cnt <= 'd0;
                        wr_bank <= (wr_bank + 2'd1) % 3'd3;  // 切换 bank
                        if (frame_end) begin
                            row_cnt    <= 'd0;
                            frame_done <= 1'b1;
                        end else begin
                            row_cnt <= row_cnt + 1'd1;
                        end
                    end else begin
                        col_cnt <= col_cnt + 1'd1;
                    end

                    // 如果窗口有效，进入输出状态
                    if (window_valid) begin
                        state         <= OUTPUT;
                        s_axis_tready <= 1'b0;
                        out_ch_cnt    <= 'd0;
                    end
                end
            end

            //--------------------------------------------------------
            OUTPUT: begin
                s_axis_tready <= 1'b0;

                // 第一个周期：tvalid 还没拉高，先置 1 并输出第一个数据
                if (out_reg_valid && !m_axis_tvalid) begin
                    m_axis_tvalid <= 1'b1;
                    m_axis_tdata  <= out_reg[out_ch_cnt];
                    m_axis_tlast  <= (out_pixel_cnt == OUT_PIXEL_CNT - 1);
                end

                // 真正握手成功（tvalid && tready）时才计数、更新数据
                if (m_axis_tvalid && m_axis_tready) begin
                    out_pixel_cnt <= out_pixel_cnt + 1'd1;

                    if (out_ch_cnt == OUT_CH - 1) begin
                        // 6 通道全部输出完毕
                        out_ch_cnt    <= 'd0;
                        m_axis_tvalid <= 1'b0;
                        // out_reg_valid 清0由独立always块处理（避免多驱动）
                        if (frame_done) begin
                            state      <= INPUT;
                            frame_done <= 1'b0;
                        end else begin
                            state         <= INPUT;
                            s_axis_tready <= 1'b1;
                        end
                    end else begin
                        // 更新到下一个通道
                        out_ch_cnt   <= out_ch_cnt + 1'd1;
                        m_axis_tdata <= out_reg[out_ch_cnt + 1'd1];
                        // 下一拍若为最后一个输出，提前拉高 TLAST
                        m_axis_tlast <= (out_pixel_cnt == OUT_PIXEL_CNT - 2);
                    end
                end
            end

            default: state <= IDLE;
        endcase
    end
end

endmodule

//=============================================================================
// 子模块：算术右移 + 饱和截断
//=============================================================================
module conv_saturate #(
    parameter IN_W       = 32,
    parameter OUT_W      = 8,
    parameter SHIFT_BITS = 4
)(
    input  wire signed [IN_W-1:0]  data_in,
    output reg  signed [OUT_W-1:0] data_out
);
    wire signed [IN_W-1:0] shifted;
    assign shifted = data_in >>> SHIFT_BITS;  // 算术右移

    always @(*) begin
        if (shifted > $signed({{(IN_W-OUT_W+1){1'b0}}, {(OUT_W-1){1'b1}}}))
            data_out = {1'b0, {(OUT_W-1){1'b1}}};   // 正饱和 = +127
        else if (shifted < $signed({{(IN_W-OUT_W+1){1'b1}}, {(OUT_W-1){1'b0}}}))
            data_out = {1'b1, {(OUT_W-1){1'b0}}};   // 负饱和 = -128
        else
            data_out = shifted[OUT_W-1:0];
    end
endmodule
