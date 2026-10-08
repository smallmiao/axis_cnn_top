//=============================================================================
// 批量归一化加速器 (axis_bn)
// 推理阶段简化为逐通道线性变换：y = x * alpha + beta
// 其中 alpha = scale / sqrt(var + eps), beta = bias - mean * alpha
// AXI-Stream 接口，通道优先输入，一级寄存器打拍输出
//=============================================================================
module axis_bn #(
    parameter CHANNELS   = 6,     // 通道数
    parameter DATA_W     = 8,     // 输入数据位宽（有符号）
    parameter ALPHA_W    = 8,     // alpha 位宽（有符号，定点缩放）
    parameter BETA_W     = 16,    // beta 位宽（有符号）
    parameter ACC_W      = 32,    // 累加器位宽
    parameter SHIFT_BITS = 4,     // 算术右移位数
    parameter OUT_W      = 8      // 输出数据位宽（有符号）
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //==================== 输入 AXI-Stream ====================
    input  wire signed [DATA_W-1:0]   s_axis_tdata,
    input  wire                        s_axis_tvalid,
    output wire                        s_axis_tready,
    input  wire                        s_axis_tlast,

    //==================== 输出 AXI-Stream ====================
    output reg signed [OUT_W-1:0]     m_axis_tdata,
    output reg                         m_axis_tvalid,
    input  wire                         m_axis_tready,
    output reg                         m_axis_tlast,

    //==================== alpha 配置接口 ====================
    input  wire                         alpha_wr_en,
    input  wire [$clog2(CHANNELS)-1:0] alpha_ch,
    input  wire signed [ALPHA_W-1:0]   alpha_data,

    //==================== beta 配置接口 ====================
    input  wire                         beta_wr_en,
    input  wire [$clog2(CHANNELS)-1:0] beta_ch,
    input  wire signed [BETA_W-1:0]    beta_data
);

//=============================================================================
// 本地参数
//=============================================================================
localparam CH_W    = $clog2(CHANNELS);
localparam MUL_W   = DATA_W + ALPHA_W;  // 乘法结果位宽 16

//=============================================================================
// 内部存储
//=============================================================================
reg signed [ALPHA_W-1:0] alpha_ram [0:CHANNELS-1];
reg signed [BETA_W-1:0]  beta_ram  [0:CHANNELS-1];

//=============================================================================
// 通道计数器（通道优先输入，每 CHANNELS 个数据循环一次）
//=============================================================================
reg [CH_W-1:0] ch_cnt;

//=============================================================================
// 计算通路
//=============================================================================
wire signed [MUL_W-1:0]   product;     // 乘法结果
wire signed [ACC_W-1:0]   product_ext; // 乘法结果符号扩展
wire signed [ACC_W-1:0]   beta_ext;    // beta 符号扩展
wire signed [ACC_W-1:0]   acc_next;    // 累加结果（组合逻辑）
wire signed [ACC_W-1:0]   shifted;     // 右移结果
reg signed [OUT_W-1:0]    sat_out;     // 饱和截断结果

// 乘法：输入 × 当前通道 alpha
assign product = s_axis_tdata * alpha_ram[ch_cnt];
assign product_ext = {{(ACC_W-MUL_W){product[MUL_W-1]}}, product};
assign beta_ext = {{(ACC_W-BETA_W){beta_ram[ch_cnt][BETA_W-1]}}, beta_ram[ch_cnt]};

// 累加：x * alpha + beta
assign acc_next = product_ext + beta_ext;

// 算术右移
assign shifted = acc_next >>> SHIFT_BITS;

//=============================================================================
// 饱和截断
//=============================================================================
always @(*) begin
    if (shifted > $signed({{(ACC_W-OUT_W+1){1'b0}}, {(OUT_W-1){1'b1}}})) begin
        sat_out = {1'b0, {(OUT_W-1){1'b1}}};  // +127
    end else if (shifted < $signed({{(ACC_W-OUT_W+1){1'b1}}, {(OUT_W-1){1'b0}}})) begin
        sat_out = {1'b1, {(OUT_W-1){1'b0}}};  // -128
    end else begin
        sat_out = shifted[OUT_W-1:0];
    end
end

//=============================================================================
// alpha/beta 配置（写优先，随时可写）
//=============================================================================
always @(posedge clk) begin
    if (alpha_wr_en) begin
        alpha_ram[alpha_ch] <= alpha_data;
    end
    if (beta_wr_en) begin
        beta_ram[beta_ch] <= beta_data;
    end
end

//=============================================================================
// 一级寄存器打拍输出 + 通道计数
//=============================================================================
assign s_axis_tready = 1'b1;   // 恒就绪（Testbench 场景下下游恒就绪）

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        m_axis_tvalid <= 1'b0;
        m_axis_tdata  <= 'd0;
        m_axis_tlast  <= 1'b0;
        ch_cnt        <= 'd0;
    end else begin
        // 输出打拍
        m_axis_tvalid <= s_axis_tvalid;
        m_axis_tdata  <= sat_out;
        m_axis_tlast  <= s_axis_tlast;

        // 通道计数：握手成功时增加
        if (s_axis_tvalid && s_axis_tready) begin
            if (ch_cnt == CHANNELS - 1) begin
                ch_cnt <= 'd0;
            end else begin
                ch_cnt <= ch_cnt + 1'd1;
            end
        end
    end
end

endmodule
