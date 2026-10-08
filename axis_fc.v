//=============================================================================
// 全连接层加速器 (axis_fc)
// AXI-Stream 接口，串行累加实现矩阵向量乘 + 偏置 + 定点量化
// 输入 IN_WIDTH 个数据 → 输出 OUT_WIDTH 个数据
// 量化：32bit 累加 → 算术右移 SHIFT_BITS → 饱和截断到 OUT_W bit
//=============================================================================
module axis_fc #(
    parameter IN_WIDTH   = 54,    // 输入维度（3x3x6）
    parameter OUT_WIDTH  = 10,    // 输出维度（分类数）
    parameter DATA_W     = 8,     // 输入/权重位宽（有符号）
    parameter BIAS_W     = 16,    // 偏置位宽（有符号）
    parameter ACC_W      = 32,    // 累加器位宽
    parameter SHIFT_BITS = 4,     // 算术右移位数
    parameter OUT_W      = 8      // 输出位宽（有符号）
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //==================== 输入 AXI-Stream ====================
    input  wire signed [DATA_W-1:0]   s_axis_tdata,
    input  wire                        s_axis_tvalid,
    output reg                         s_axis_tready,
    input  wire                        s_axis_tlast,

    //==================== 输出 AXI-Stream ====================
    output reg signed [OUT_W-1:0]    m_axis_tdata,
    output reg                        m_axis_tvalid,
    input  wire                        m_axis_tready,
    output reg                        m_axis_tlast,

    //==================== 权重配置接口 ====================
    input  wire                        weight_wr_en,
    input  wire [$clog2(OUT_WIDTH)-1:0] weight_out_idx,
    input  wire [$clog2(IN_WIDTH)-1:0]  weight_in_idx,
    input  wire signed [DATA_W-1:0]     weight_data,

    //==================== 偏置配置接口 ====================
    input  wire                        bias_wr_en,
    input  wire [$clog2(OUT_WIDTH)-1:0] bias_out_idx,
    input  wire signed [BIAS_W-1:0]     bias_data
);

//=============================================================================
// 本地参数
//=============================================================================
localparam IN_IDX_W   = $clog2(IN_WIDTH);
localparam OUT_IDX_W  = $clog2(OUT_WIDTH);
localparam MUL_W      = DATA_W * 2;  // 乘法结果位宽 16
localparam signed [OUT_W-1:0] OUT_MAX = 8'sd127;
localparam signed [OUT_W-1:0] OUT_MIN = -8'sd128;

//=============================================================================
// 状态机
//=============================================================================
localparam IDLE    = 2'd0;
localparam INPUT   = 2'd1;
localparam COMPUTE = 2'd2;
localparam OUTPUT  = 2'd3;
reg [1:0] state;

//=============================================================================
// 内部存储
//=============================================================================
reg signed [DATA_W-1:0]  input_ram  [0:IN_WIDTH-1];
reg signed [DATA_W-1:0]  weight_ram [0:OUT_WIDTH-1][0:IN_WIDTH-1];
reg signed [BIAS_W-1:0]  bias_ram   [0:OUT_WIDTH-1];
reg signed [OUT_W-1:0]   output_reg [0:OUT_WIDTH-1];

//=============================================================================
// 计数器
//=============================================================================
reg [IN_IDX_W-1:0]  in_idx;      // 输入索引（接收/计算用）
reg [OUT_IDX_W-1:0] out_idx;     // 输出索引（计算/输出用）
reg [IN_IDX_W-1:0]  input_cnt;   // 已接收输入计数

//=============================================================================
// 计算通路
//=============================================================================
reg signed [ACC_W-1:0]  acc;          // 累加器
wire signed [MUL_W-1:0]  product;      // 乘法结果
wire signed [ACC_W-1:0]  product_ext;  // 乘法结果符号扩展
wire signed [ACC_W-1:0]  bias_ext;     // 偏置符号扩展
wire signed [ACC_W-1:0]  shifted;      // 右移结果
reg signed [OUT_W-1:0]   sat_out;      // 饱和截断结果

// 乘法：输入 × 权重
assign product = input_ram[in_idx] * weight_ram[out_idx][in_idx];
assign product_ext = $signed({{(ACC_W-MUL_W){product[MUL_W-1]}}, product});
assign bias_ext = $signed({{(ACC_W-BIAS_W){bias_ram[out_idx][BIAS_W-1]}}, bias_ram[out_idx]});

// 预计算下一个周期的累加值（包含当前乘加）
// in_idx==0: 偏置 + 第一个乘加；in_idx!=0: 累加 + 当前乘加
wire signed [ACC_W-1:0] acc_next;
assign acc_next = (in_idx == 0) ? (bias_ext + product_ext) : (acc + product_ext);

// 算术右移（基于 acc_next，确保包含所有乘加）
assign shifted = acc_next >>> SHIFT_BITS;

//=============================================================================
// 饱和截断
//=============================================================================
always @(*) begin
    if (shifted > OUT_MAX) begin
        sat_out = OUT_MAX;  // +127
    end else if (shifted < OUT_MIN) begin
        sat_out = OUT_MIN;  // -128
    end else begin
        sat_out = shifted[OUT_W-1:0];
    end
end

//=============================================================================
// 权重/偏置配置（写优先，随时可写）
//=============================================================================
always @(posedge clk) begin
    if (weight_wr_en) begin
        weight_ram[weight_out_idx][weight_in_idx] <= weight_data;
    end
    if (bias_wr_en) begin
        bias_ram[bias_out_idx] <= bias_data;
    end
end

//=============================================================================
// 主状态机 + 数据通路
//=============================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state         <= IDLE;
        in_idx        <= 'd0;
        out_idx       <= 'd0;
        input_cnt     <= 'd0;
        acc           <= 'd0;
        s_axis_tready <= 1'b0;
        m_axis_tvalid <= 1'b0;
        m_axis_tdata  <= 'd0;
        m_axis_tlast  <= 1'b0;
    end else begin
        case (state)
            //=================================================================
            // IDLE：等待启动（配置完成后进入 INPUT）
            //=================================================================
            IDLE: begin
                s_axis_tready <= 1'b1;
                m_axis_tvalid <= 1'b0;
                m_axis_tlast  <= 1'b0;
                in_idx        <= 'd0;
                out_idx       <= 'd0;
                if (s_axis_tvalid && s_axis_tready) begin
                    // 存入第一个数据（避免 IDLE 周期丢弃第一个输入）
                    input_ram[input_cnt] <= s_axis_tdata;
                    input_cnt <= input_cnt + 1'd1;
                    state <= INPUT;
                end
            end

            //=================================================================
            // INPUT：接收输入向量，存入 input_ram
            //=================================================================
            INPUT: begin
                s_axis_tready <= 1'b1;
                if (s_axis_tvalid && s_axis_tready) begin
                    input_ram[input_cnt] <= s_axis_tdata;
                    input_cnt <= input_cnt + 1'd1;
                    if (input_cnt == IN_WIDTH - 1) begin
                        state     <= COMPUTE;
                        in_idx    <= 'd0;
                        out_idx   <= 'd0;
                        acc       <= 'd0;
                    end
                end
            end

            //=================================================================
            // COMPUTE：串行累加计算每个输出通道
            //=================================================================
            COMPUTE: begin
                s_axis_tready <= 1'b0;

                // 累加器更新（与 acc_next 一致，包含当前乘加）
                acc <= acc_next;

                if (in_idx == IN_WIDTH - 1) begin
                    // 当前输出通道计算完成，右移+饱和截断，存入输出寄存器
                    // sat_out 基于 acc_next（完整累加结果）
                    output_reg[out_idx] <= sat_out;

                    in_idx <= 'd0;
                    if (out_idx == OUT_WIDTH - 1) begin
                        // 所有输出通道计算完成，进入输出状态
                        state   <= OUTPUT;
                        out_idx <= 'd0;
                    end else begin
                        out_idx <= out_idx + 1'd1;
                    end
                end else begin
                    in_idx <= in_idx + 1'd1;
                end
            end

            //=================================================================
            // OUTPUT：标准 AXI-Stream 主接口握手输出
            //=================================================================
            OUTPUT: begin
                if (!m_axis_tvalid) begin
                    m_axis_tvalid <= 1'b1;
                    m_axis_tdata  <= output_reg[out_idx];
                    m_axis_tlast  <= (out_idx == OUT_WIDTH - 1) ? 1'b1 : 1'b0;
                end

                if (m_axis_tvalid && m_axis_tready) begin
                    if (out_idx == OUT_WIDTH - 1) begin
                        state         <= IDLE;
                        m_axis_tvalid <= 1'b0;
                        m_axis_tlast  <= 1'b0;
                        out_idx       <= 'd0;
                    end else begin
                        out_idx       <= out_idx + 1'd1;
                        m_axis_tvalid <= 1'b0;
                    end
                end
            end

            default: state <= IDLE;
        endcase
    end
end

endmodule
