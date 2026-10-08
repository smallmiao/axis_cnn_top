//=============================================================================
// cnn_axis_ip.v - Zynq AXI-Stream IP Wrapper for axis_cnn_top
// Auto-loads weights from hex files after reset
//=============================================================================
module cnn_axis_ip #(
    parameter DATA_WIDTH = 8
)(
    input  wire                          aclk,
    input  wire                          aresetn,

    input  wire [DATA_WIDTH-1:0]         s_axis_tdata,
    input  wire                          s_axis_tvalid,
    output wire                          s_axis_tready,
    input  wire                          s_axis_tlast,

    output wire [DATA_WIDTH-1:0]         m_axis_tdata,
    output wire                          m_axis_tvalid,
    input  wire                          m_axis_tready,
    output wire                          m_axis_tlast
);

    //=============================================================================
    // Weight ROM
    //=============================================================================
    reg [7:0]  conv_weight_rom [0:53];
    reg [15:0] conv_bias_rom   [0:5];
    reg [7:0]  bn_alpha_rom    [0:5];
    reg [15:0] bn_beta_rom     [0:5];
    reg [7:0]  fc_weight_rom   [0:539];
    reg [15:0] fc_bias_rom     [0:9];

    initial begin
        $readmemh("F:/20260822simulation/axis_cnn_top/conv_weights.hex", conv_weight_rom);
        $readmemh("F:/20260822simulation/axis_cnn_top/conv_biases.hex",  conv_bias_rom);
        $readmemh("F:/20260822simulation/axis_cnn_top/bn_alpha.hex",     bn_alpha_rom);
        $readmemh("F:/20260822simulation/axis_cnn_top/bn_beta.hex",      bn_beta_rom);
        $readmemh("F:/20260822simulation/axis_cnn_top/fc_weights.hex",   fc_weight_rom);
        $readmemh("F:/20260822simulation/axis_cnn_top/fc_biases.hex",    fc_bias_rom);
    end

    //=============================================================================
    // Weight load state machine
    //=============================================================================
    localparam S_CONV_W=0, S_CONV_B=1, S_BN_A=2, S_BN_B=3, S_FC_W=4, S_FC_B=5, S_DONE=6;
    reg [2:0]  wstate;
    reg [9:0]  wcnt;

    wire [2:0]  conv_weight_out_ch = wcnt / 9;
    wire [3:0]  conv_weight_addr   = wcnt % 9;
    wire [7:0]  conv_weight_data   = conv_weight_rom[wcnt];
    wire        conv_weight_wr_en  = (wstate == S_CONV_W);
    wire [2:0]  conv_bias_out_ch   = wcnt[2:0];
    wire [15:0] conv_bias_data     = conv_bias_rom[wcnt[2:0]];
    wire        conv_bias_wr_en    = (wstate == S_CONV_B);

    wire [2:0]  bn_alpha_ch        = wcnt[2:0];
    wire [7:0]  bn_alpha_data      = bn_alpha_rom[wcnt[2:0]];
    wire        bn_alpha_wr_en     = (wstate == S_BN_A);
    wire [2:0]  bn_beta_ch         = wcnt[2:0];
    wire [15:0] bn_beta_data       = bn_beta_rom[wcnt[2:0]];
    wire        bn_beta_wr_en      = (wstate == S_BN_B);

    wire [3:0]  fc_weight_out_idx  = wcnt / 54;
    wire [5:0]  fc_weight_in_idx   = wcnt % 54;
    wire [7:0]  fc_weight_data     = fc_weight_rom[wcnt];
    wire        fc_weight_wr_en    = (wstate == S_FC_W);
    wire [3:0]  fc_bias_out_idx    = wcnt[3:0];
    wire [15:0] fc_bias_data       = fc_bias_rom[wcnt[3:0]];
    wire        fc_bias_wr_en      = (wstate == S_FC_B);

    always @(posedge aclk) begin
        if (!aresetn) begin
            wstate <= S_CONV_W;
            wcnt   <= 0;
        end else begin
            case (wstate)
                S_CONV_W: begin
                    if (wcnt == 53) begin wstate <= S_CONV_B; wcnt <= 0; end
                    else wcnt <= wcnt + 1;
                end
                S_CONV_B: begin
                    if (wcnt == 5) begin wstate <= S_BN_A; wcnt <= 0; end
                    else wcnt <= wcnt + 1;
                end
                S_BN_A: begin
                    if (wcnt == 5) begin wstate <= S_BN_B; wcnt <= 0; end
                    else wcnt <= wcnt + 1;
                end
                S_BN_B: begin
                    if (wcnt == 5) begin wstate <= S_FC_W; wcnt <= 0; end
                    else wcnt <= wcnt + 1;
                end
                S_FC_W: begin
                    if (wcnt == 539) begin wstate <= S_FC_B; wcnt <= 0; end
                    else wcnt <= wcnt + 1;
                end
                S_FC_B: begin
                    if (wcnt == 9) wstate <= S_DONE;
                    else wcnt <= wcnt + 1;
                end
                default: wstate <= S_DONE;
            endcase
        end
    end

    //=============================================================================
    // CNN core
    //=============================================================================
    axis_cnn_top u_cnn (
        .clk(aclk),
        .rst_n(aresetn),

        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),

        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast),

        .conv_weight_wr_en(conv_weight_wr_en),
        .conv_weight_out_ch(conv_weight_out_ch),
        .conv_weight_addr(conv_weight_addr),
        .conv_weight_data(conv_weight_data),

        .conv_bias_wr_en(conv_bias_wr_en),
        .conv_bias_out_ch(conv_bias_out_ch),
        .conv_bias_data(conv_bias_data),

        .bn_alpha_wr_en(bn_alpha_wr_en),
        .bn_alpha_ch(bn_alpha_ch),
        .bn_alpha_data(bn_alpha_data),

        .bn_beta_wr_en(bn_beta_wr_en),
        .bn_beta_ch(bn_beta_ch),
        .bn_beta_data(bn_beta_data),

        .fc_weight_wr_en(fc_weight_wr_en),
        .fc_weight_out_idx(fc_weight_out_idx),
        .fc_weight_in_idx(fc_weight_in_idx),
        .fc_weight_data(fc_weight_data),

        .fc_bias_wr_en(fc_bias_wr_en),
        .fc_bias_out_idx(fc_bias_out_idx),
        .fc_bias_data(fc_bias_data)
    );

endmodule
