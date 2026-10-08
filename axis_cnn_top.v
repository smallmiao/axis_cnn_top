//=============================================================================
// 完整 CNN 推理链路顶层 (axis_cnn_top)
// Conv(3x3, 1->6ch) -> BN -> Pool(2x2) -> ReLU -> FC(54->10)
// 输入: 8x8x1 = 64 -> 输出: 10 分类结果
//=============================================================================
module axis_cnn_top #(
    //==================== 卷积参数 ====================
    parameter CONV_IN_CH      = 1,
    parameter CONV_OUT_CH     = 6,
    parameter CONV_DATA_W     = 8,
    parameter CONV_WEIGHT_W   = 8,
    parameter CONV_BIAS_W     = 16,
    parameter CONV_ACC_W      = 32,
    parameter CONV_OUT_W      = 8,
    parameter CONV_SHIFT_BITS = 4,
    parameter CONV_IMG_W      = 8,
    parameter CONV_IMG_H      = 8,
    //==================== BN 参数 ====================
    parameter BN_CHANNELS     = 6,
    parameter BN_DATA_W       = 8,
    parameter BN_ALPHA_W      = 8,
    parameter BN_BETA_W       = 16,
    parameter BN_ACC_W        = 32,
    parameter BN_SHIFT_BITS   = 4,
    parameter BN_OUT_W        = 8,
    //==================== 池化参数 ====================
    parameter POOL_CH         = 6,
    parameter POOL_DATA_W     = 8,
    parameter POOL_IMG_W      = 6,
    parameter POOL_IMG_H      = 6,
    //==================== ReLU 参数 ====================
    parameter RELU_DATA_W     = 8,
    //==================== FC 参数 ====================
    parameter FC_IN_WIDTH     = 54,    // 3x3x6
    parameter FC_OUT_WIDTH    = 10,
    parameter FC_DATA_W       = 8,
    parameter FC_BIAS_W       = 16,
    parameter FC_ACC_W        = 32,
    parameter FC_SHIFT_BITS   = 4,
    parameter FC_OUT_W        = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //==================== 输入 AXI-Stream ====================
    input  wire signed [CONV_DATA_W-1:0]  s_axis_tdata,
    input  wire                             s_axis_tvalid,
    output wire                             s_axis_tready,
    input  wire                             s_axis_tlast,

    //==================== 输出 AXI-Stream ====================
    output wire signed [FC_OUT_W-1:0]     m_axis_tdata,
    output wire                             m_axis_tvalid,
    input  wire                             m_axis_tready,
    output wire                             m_axis_tlast,

    //==================== 卷积权重配置 ====================
    input  wire                             conv_weight_wr_en,
    input  wire [$clog2(CONV_OUT_CH)-1:0]  conv_weight_out_ch,
    input  wire [3:0]                       conv_weight_addr,
    input  wire signed [CONV_WEIGHT_W-1:0] conv_weight_data,

    //==================== 卷积偏置配置 ====================
    input  wire                             conv_bias_wr_en,
    input  wire [$clog2(CONV_OUT_CH)-1:0]  conv_bias_out_ch,
    input  wire signed [CONV_BIAS_W-1:0]   conv_bias_data,

    //==================== BN alpha 配置 ====================
    input  wire                             bn_alpha_wr_en,
    input  wire [$clog2(BN_CHANNELS)-1:0]  bn_alpha_ch,
    input  wire signed [BN_ALPHA_W-1:0]    bn_alpha_data,

    //==================== BN beta 配置 ====================
    input  wire                             bn_beta_wr_en,
    input  wire [$clog2(BN_CHANNELS)-1:0]  bn_beta_ch,
    input  wire signed [BN_BETA_W-1:0]     bn_beta_data,

    //==================== FC权重配置 ====================
    input  wire                             fc_weight_wr_en,
    input  wire [$clog2(FC_OUT_WIDTH)-1:0]  fc_weight_out_idx,
    input  wire [$clog2(FC_IN_WIDTH)-1:0]   fc_weight_in_idx,
    input  wire signed [FC_DATA_W-1:0]      fc_weight_data,

    //==================== FC偏置配置 ====================
    input  wire                             fc_bias_wr_en,
    input  wire [$clog2(FC_OUT_WIDTH)-1:0]  fc_bias_out_idx,
    input  wire signed [FC_BIAS_W-1:0]      fc_bias_data
);

//=============================================================================
// 内部 AXI-Stream 连线
//=============================================================================
// Conv -> BN
wire signed [CONV_OUT_W-1:0]  conv_to_bn_tdata;
wire                            conv_to_bn_tvalid;
wire                            conv_to_bn_tready;
wire                            conv_to_bn_tlast;

// BN -> Pool
wire signed [BN_OUT_W-1:0]    bn_to_pool_tdata;
wire                            bn_to_pool_tvalid;
wire                            bn_to_pool_tready;
wire                            bn_to_pool_tlast;

// Pool -> ReLU
wire signed [POOL_DATA_W-1:0]  pool_to_relu_tdata;
wire                            pool_to_relu_tvalid;
wire                            pool_to_relu_tready;
wire                            pool_to_relu_tlast;

// ReLU -> FC
wire signed [RELU_DATA_W-1:0]  relu_to_fc_tdata;
wire                            relu_to_fc_tvalid;
wire                            relu_to_fc_tready;
wire                            relu_to_fc_tlast;

//=============================================================================
// 例化：多通道3x3卷积
//=============================================================================
axis_conv3x3_mc #(
    .IN_CH(CONV_IN_CH),
    .OUT_CH(CONV_OUT_CH),
    .DATA_W(CONV_DATA_W),
    .WEIGHT_W(CONV_WEIGHT_W),
    .BIAS_W(CONV_BIAS_W),
    .ACC_W(CONV_ACC_W),
    .OUT_W(CONV_OUT_W),
    .SHIFT_BITS(CONV_SHIFT_BITS),
    .IMG_W(CONV_IMG_W),
    .IMG_H(CONV_IMG_H)
) u_conv (
    .clk(clk),
    .rst_n(rst_n),
    .s_axis_tdata(s_axis_tdata),
    .s_axis_tvalid(s_axis_tvalid),
    .s_axis_tready(s_axis_tready),
    .s_axis_tlast(s_axis_tlast),
    .m_axis_tdata(conv_to_bn_tdata),
    .m_axis_tvalid(conv_to_bn_tvalid),
    .m_axis_tready(conv_to_bn_tready),
    .m_axis_tlast(conv_to_bn_tlast),
    .weight_wr_en(conv_weight_wr_en),
    .weight_out_ch(conv_weight_out_ch),
    .weight_addr(conv_weight_addr),
    .weight_data(conv_weight_data),
    .bias_wr_en(conv_bias_wr_en),
    .bias_out_ch(conv_bias_out_ch),
    .bias_data(conv_bias_data)
);

//=============================================================================
// 例化：批量归一化 BN
//=============================================================================
axis_bn #(
    .CHANNELS(BN_CHANNELS),
    .DATA_W(BN_DATA_W),
    .ALPHA_W(BN_ALPHA_W),
    .BETA_W(BN_BETA_W),
    .ACC_W(BN_ACC_W),
    .SHIFT_BITS(BN_SHIFT_BITS),
    .OUT_W(BN_OUT_W)
) u_bn (
    .clk(clk),
    .rst_n(rst_n),
    .s_axis_tdata(conv_to_bn_tdata),
    .s_axis_tvalid(conv_to_bn_tvalid),
    .s_axis_tready(conv_to_bn_tready),
    .s_axis_tlast(conv_to_bn_tlast),
    .m_axis_tdata(bn_to_pool_tdata),
    .m_axis_tvalid(bn_to_pool_tvalid),
    .m_axis_tready(bn_to_pool_tready),
    .m_axis_tlast(bn_to_pool_tlast),
    .alpha_wr_en(bn_alpha_wr_en),
    .alpha_ch(bn_alpha_ch),
    .alpha_data(bn_alpha_data),
    .beta_wr_en(bn_beta_wr_en),
    .beta_ch(bn_beta_ch),
    .beta_data(bn_beta_data)
);

//=============================================================================
// 例化：多通道2x2最大池化
//=============================================================================
axis_maxpool2x2 #(
    .CHANNELS(POOL_CH),
    .DATA_W(POOL_DATA_W),
    .IMG_W(POOL_IMG_W),
    .IMG_H(POOL_IMG_H)
) u_pool (
    .clk(clk),
    .rst_n(rst_n),
    .s_axis_tdata(bn_to_pool_tdata),
    .s_axis_tvalid(bn_to_pool_tvalid),
    .s_axis_tready(bn_to_pool_tready),
    .s_axis_tlast(bn_to_pool_tlast),
    .m_axis_tdata(pool_to_relu_tdata),
    .m_axis_tvalid(pool_to_relu_tvalid),
    .m_axis_tready(pool_to_relu_tready),
    .m_axis_tlast(pool_to_relu_tlast)
);

//=============================================================================
// 例化：ReLU 激活函数
//=============================================================================
axis_relu #(
    .DATA_W(RELU_DATA_W)
) u_relu (
    .clk(clk),
    .rst_n(rst_n),
    .s_axis_tdata(pool_to_relu_tdata),
    .s_axis_tvalid(pool_to_relu_tvalid),
    .s_axis_tready(pool_to_relu_tready),
    .s_axis_tlast(pool_to_relu_tlast),
    .m_axis_tdata(relu_to_fc_tdata),
    .m_axis_tvalid(relu_to_fc_tvalid),
    .m_axis_tready(relu_to_fc_tready),
    .m_axis_tlast(relu_to_fc_tlast)
);

//=============================================================================
// 例化：全连接层
//=============================================================================
axis_fc #(
    .IN_WIDTH(FC_IN_WIDTH),
    .OUT_WIDTH(FC_OUT_WIDTH),
    .DATA_W(FC_DATA_W),
    .BIAS_W(FC_BIAS_W),
    .ACC_W(FC_ACC_W),
    .SHIFT_BITS(FC_SHIFT_BITS),
    .OUT_W(FC_OUT_W)
) u_fc (
    .clk(clk),
    .rst_n(rst_n),
    .s_axis_tdata(relu_to_fc_tdata),
    .s_axis_tvalid(relu_to_fc_tvalid),
    .s_axis_tready(relu_to_fc_tready),
    .s_axis_tlast(relu_to_fc_tlast),
    .m_axis_tdata(m_axis_tdata),
    .m_axis_tvalid(m_axis_tvalid),
    .m_axis_tready(m_axis_tready),
    .m_axis_tlast(m_axis_tlast),
    .weight_wr_en(fc_weight_wr_en),
    .weight_out_idx(fc_weight_out_idx),
    .weight_in_idx(fc_weight_in_idx),
    .weight_data(fc_weight_data),
    .bias_wr_en(fc_bias_wr_en),
    .bias_out_idx(fc_bias_out_idx),
    .bias_data(fc_bias_data)
);

endmodule
