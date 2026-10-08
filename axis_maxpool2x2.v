//=============================================================================
// 多通道 2x2 最大池化加速器 (axis_maxpool2x2)
// 支持参数化通道数，输入/输出均为 AXI-Stream 接口
// 输入数据按通道优先排列（每像素位置 CHANNELS 个数据）
// 输出同样按通道优先排列
// 池化窗口 2x2，步长 2，无 padding
//=============================================================================
module axis_maxpool2x2 #(
    parameter CHANNELS = 6,    // 通道数
    parameter DATA_W   = 8,    // 数据位宽（有符号）
    parameter IMG_W    = 6,    // 输入图像宽度
    parameter IMG_H    = 6     // 输入图像高度
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //==================== 输入 AXI-Stream ====================
    input  wire signed [DATA_W-1:0]  s_axis_tdata,
    input  wire                        s_axis_tvalid,
    output reg                         s_axis_tready,
    input  wire                        s_axis_tlast,

    //==================== 输出 AXI-Stream ====================
    output reg signed [DATA_W-1:0]  m_axis_tdata,
    output reg                        m_axis_tvalid,
    input  wire                        m_axis_tready,
    output reg                        m_axis_tlast
);

//=============================================================================
// 本地参数
//=============================================================================
localparam OUT_IMG_W     = IMG_W / 2;                          // 输出宽度
localparam OUT_IMG_H     = IMG_H / 2;                          // 输出高度
localparam OUT_PIXELS    = OUT_IMG_W * OUT_IMG_H;              // 输出窗口数
localparam COL_W          = $clog2(IMG_W);                      // 列计数器位宽
localparam ROW_W          = $clog2(IMG_H);                      // 行计数器位宽
localparam CH_W           = $clog2(CHANNELS);                   // 通道计数器位宽
localparam OUT_PIXEL_CNT  = CHANNELS * OUT_IMG_W * OUT_IMG_H;  // 总输出像素数

//=============================================================================
// 内部信号
//=============================================================================
reg [COL_W-1:0]  col_cnt;          // 当前输入列计数器
reg [ROW_W-1:0]  row_cnt;          // 当前输入行计数器
reg [CH_W-1:0]   ch_cnt;           // 当前输入通道计数器
reg [CH_W-1:0]   out_ch_cnt;       // 输出通道计数器
reg [15:0]       out_pixel_cnt;     // 已输出像素计数，供 tlast 处理
reg              frame_done;       // 一帧输入完成标志
reg [15:0]       pool_idx;          // 当前累计已生成的 2x2 池化输出像素数
reg [15:0]       out_idx;           // 当前输出索引

// 状态机
localparam IDLE     = 2'd0;
localparam INPUT    = 2'd1;
localparam OUTPUT   = 2'd2;
reg [1:0] state;

wire pool_valid;    // 2x2 池化窗口有效（当前位置是窗口右下角）
wire line_end;      // 一行结束
wire frame_end;     // 一帧结束

//=============================================================================
// Line Buffer: 2 个行 bank 轮转，存储上一行 + 当前行
// 每个 bank 存储 IMG_W 列 × CHANNELS 通道
//=============================================================================
reg signed [DATA_W-1:0] line_buf [0:1][0:IMG_W-1][0:CHANNELS-1];
reg wr_bank;    // 当前写入 bank (0/1 轮转)
wire prev_bank = ~wr_bank;  // 上一行 bank

//=============================================================================
// 各通道最大值寄存器 + 已完成的窗口输出缓存
// max_val[0-4] 用寄存器（ch0-ch4 时赋值，到 ch5 时已生效）
// max_val[5] 因 ch5 周期非阻塞赋值延迟，out_buf[5] 直接用组合逻辑计算
//=============================================================================
reg signed [DATA_W-1:0] max_val [0:CHANNELS-1];
reg signed [DATA_W-1:0] out_buf [0:OUT_PIXEL_CNT-1][0:CHANNELS-1];

// 组合逻辑：当前通道（ch_cnt）的 2x2 最大值，用于 out_buf[5] 避免非阻塞延迟
wire signed [DATA_W-1:0] cur_ch_max = (row_cnt >= 1 && col_cnt >= 1) ?
    max4(
        line_buf[prev_bank][col_cnt - 1][ch_cnt],  // 左上
        line_buf[prev_bank][col_cnt][ch_cnt],       // 右上
        line_buf[wr_bank][col_cnt - 1][ch_cnt],     // 左下
        s_axis_tdata                                  // 右下
    ) : 'd0;

//=============================================================================
// 组合逻辑：池化窗口有效判断
// 当 row>=1, col>=1, row为奇数, col为奇数时，当前位置是 2x2 窗口的右下角
//=============================================================================
assign pool_valid = (row_cnt >= 1) && (col_cnt >= 1) &&
                    (row_cnt[0] == 1'b1) && (col_cnt[0] == 1'b1);

assign line_end  = (col_cnt == IMG_W - 1) && (ch_cnt == CHANNELS - 1);
assign frame_end = (row_cnt == IMG_H - 1) && line_end;

//=============================================================================
// 主状态机 + 数据通路
//=============================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state         <= INPUT;
        col_cnt       <= 'd0;
        row_cnt       <= 'd0;
        ch_cnt        <= 'd0;
        out_ch_cnt    <= 'd0;
        out_pixel_cnt <= 'd0;
        out_idx       <= 'd0;
        pool_idx      <= 'd0;
        wr_bank       <= 1'b0;
        frame_done    <= 1'b0;
        s_axis_tready <= 1'b1;
        m_axis_tvalid <= 1'b0;
        m_axis_tlast  <= 1'b0;
        m_axis_tdata  <= 'd0;
    end else begin
        case (state)
            //=================================================================
            // IDLE: 软件复位/重新初始化用（正常流程不经过此状态）
            //=================================================================
            IDLE: begin
                s_axis_tready <= 1'b1;
                m_axis_tvalid <= 1'b0;
                m_axis_tlast  <= 1'b0;
                col_cnt       <= 'd0;
                row_cnt       <= 'd0;
                ch_cnt        <= 'd0;
                out_ch_cnt    <= 'd0;
                out_pixel_cnt <= 'd0;
                out_idx       <= 'd0;
                pool_idx      <= 'd0;
                wr_bank       <= 1'b0;
                frame_done    <= 1'b0;
                state         <= INPUT;
            end

            //=================================================================
            // INPUT: 接收输入并缓存整帧，再在帧结束后开始输出
            //=================================================================
            INPUT: begin
                s_axis_tready <= 1'b1;
                m_axis_tvalid <= 1'b0;
                m_axis_tlast  <= 1'b0;
                frame_done    <= 1'b0;

                if (s_axis_tvalid && s_axis_tready) begin
                    // 写入当前行 bank
                    line_buf[wr_bank][col_cnt][ch_cnt] <= s_axis_tdata;

                    // 计算当前通道的 2x2 最大值（寄存器，ch0-ch4 到 ch5 时已生效）
                    if (row_cnt >= 1 && col_cnt >= 1) begin
                        max_val[ch_cnt] <= cur_ch_max;
                    end

                    // 完成当前像素对应的一个窗口：缓存最终 6 通道结果
                    // out_buf[0-4] 用 max_val 寄存器（已生效）
                    // out_buf[5] 直接用 cur_ch_max 组合逻辑（避免 ch5 非阻塞赋值延迟）
                    if (ch_cnt == CHANNELS - 1) begin
                        if (pool_valid) begin
                            out_buf[pool_idx][0] <= max_val[0];
                            out_buf[pool_idx][1] <= max_val[1];
                            out_buf[pool_idx][2] <= max_val[2];
                            out_buf[pool_idx][3] <= max_val[3];
                            out_buf[pool_idx][4] <= max_val[4];
                            out_buf[pool_idx][5] <= cur_ch_max;
                            pool_idx <= pool_idx + 1'd1;
                        end

                        ch_cnt <= 'd0;
                        if (col_cnt == IMG_W - 1) begin
                            col_cnt <= 'd0;
                            wr_bank <= ~wr_bank;
                            if (row_cnt == IMG_H - 1) begin
                                row_cnt     <= 'd0;
                                frame_done  <= 1'b1;
                            end else begin
                                row_cnt     <= row_cnt + 1'd1;
                            end
                        end else begin
                            col_cnt <= col_cnt + 1'd1;
                        end
                    end else begin
                        ch_cnt <= ch_cnt + 1'd1;
                    end

                    if (s_axis_tlast || frame_end || frame_done) begin
                        state <= OUTPUT;
                        s_axis_tready <= 1'b0;
                        out_idx <= 'd0;
                        out_ch_cnt <= 'd0;
                        out_pixel_cnt <= 'd0;
                    end
                end
            end

            //=================================================================
            // OUTPUT: 按缓存顺序逐通道输出帧内所有池化结果
            // 标准 AXI-Stream 主接口握手：先拉高 tvalid，握手成功后更新下一个数据
            //=================================================================
            OUTPUT: begin
                if (out_idx < pool_idx) begin
                    if (!m_axis_tvalid) begin
                        m_axis_tvalid <= 1'b1;
                        m_axis_tdata  <= out_buf[out_idx][out_ch_cnt];
                        m_axis_tlast  <= ((out_idx == OUT_PIXELS - 1) &&
                                          (out_ch_cnt == CHANNELS - 1)) ? 1'b1 : 1'b0;
                    end
                end else begin
                    m_axis_tvalid <= 1'b0;
                    m_axis_tlast  <= 1'b0;
                end

                if (m_axis_tvalid && m_axis_tready) begin
                    out_pixel_cnt <= out_pixel_cnt + 1'd1;
                    if (out_ch_cnt == CHANNELS - 1) begin
                        out_ch_cnt    <= 'd0;
                        out_idx       <= out_idx + 1'd1;
                        m_axis_tvalid <= 1'b0;
                        if (out_idx == pool_idx - 1) begin
                            state         <= INPUT;
                            s_axis_tready <= 1'b1;
                            pool_idx      <= 'd0;
                            out_idx       <= 'd0;
                            out_pixel_cnt <= 'd0;
                        end
                    end else begin
                        out_ch_cnt   <= out_ch_cnt + 1'd1;
                        m_axis_tdata <= out_buf[out_idx][out_ch_cnt + 1'd1];
                        m_axis_tlast <= ((out_idx == OUT_PIXELS - 1) &&
                                         (out_ch_cnt + 1'd1 == CHANNELS - 1)) ? 1'b1 : 1'b0;
                    end
                end
            end

            default: state <= IDLE;
        endcase
    end
end

//=============================================================================
// 函数：4 输入有符号最大值
//=============================================================================
function signed [DATA_W-1:0] max4(
    input signed [DATA_W-1:0] a,
    input signed [DATA_W-1:0] b,
    input signed [DATA_W-1:0] c,
    input signed [DATA_W-1:0] d
);
    reg signed [DATA_W-1:0] tmp1, tmp2;
    begin
        tmp1 = (a > b) ? a : b;
        tmp2 = (c > d) ? c : d;
        max4 = (tmp1 > tmp2) ? tmp1 : tmp2;
    end
endfunction

endmodule
