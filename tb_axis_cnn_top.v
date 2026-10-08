//=============================================================================
// axis_cnn_top Testbench
// Conv -> BN -> Pool -> ReLU -> FC 完整 CNN 推理链路端到端验证
//=============================================================================
`timescale 1ns/1ps

module tb_axis_cnn_top;

//==================== 参数 ====================
localparam INPUT_CNT       = 64;    // 8x8x1
localparam CONV_WEIGHT_CNT = 54;    // 6ch x 9
localparam CONV_BIAS_CNT   = 6;
localparam BN_ALPHA_CNT    = 6;
localparam BN_BETA_CNT     = 6;
localparam FC_WEIGHT_CNT   = 540;   // 10ch x 54
localparam FC_BIAS_CNT     = 10;
localparam OUTPUT_CNT       = 10;

//==================== 信号 ====================
reg clk;
reg rst_n;

reg  signed [7:0]   s_axis_tdata;
reg                  s_axis_tvalid;
wire                 s_axis_tready;
reg                  s_axis_tlast;

wire signed [7:0]   m_axis_tdata;
wire                 m_axis_tvalid;
reg                  m_axis_tready;
wire                 m_axis_tlast;

// 卷积配置
reg                  conv_weight_wr_en;
reg  [2:0]           conv_weight_out_ch;
reg  [3:0]           conv_weight_addr;
reg  signed [7:0]    conv_weight_data;
reg                  conv_bias_wr_en;
reg  [2:0]           conv_bias_out_ch;
reg  signed [15:0]   conv_bias_data;

// BN 配置
reg                  bn_alpha_wr_en;
reg  [2:0]           bn_alpha_ch;
reg  signed [7:0]    bn_alpha_data;
reg                  bn_beta_wr_en;
reg  [2:0]           bn_beta_ch;
reg  signed [15:0]   bn_beta_data;

// FC 配置
reg                  fc_weight_wr_en;
reg  [3:0]           fc_weight_out_idx;
reg  [5:0]           fc_weight_in_idx;
reg  signed [7:0]    fc_weight_data;
reg                  fc_bias_wr_en;
reg  [3:0]           fc_bias_out_idx;
reg  signed [15:0]   fc_bias_data;

//==================== 存储 ====================
reg signed [7:0]    input_mem        [0:INPUT_CNT-1];
reg signed [7:0]    conv_weight_mem  [0:CONV_WEIGHT_CNT-1];
reg signed [15:0]   conv_bias_mem    [0:CONV_BIAS_CNT-1];
reg signed [7:0]    bn_alpha_mem     [0:BN_ALPHA_CNT-1];
reg signed [15:0]   bn_beta_mem      [0:BN_BETA_CNT-1];
reg signed [7:0]    fc_weight_mem    [0:FC_WEIGHT_CNT-1];
reg signed [15:0]   fc_bias_mem      [0:FC_BIAS_CNT-1];
reg signed [7:0]    output_mem       [0:OUTPUT_CNT-1];
reg signed [7:0]    golden_mem       [0:OUTPUT_CNT-1];

integer input_idx;
integer output_idx;
integer i;
integer err_cnt;
integer tlast_cnt;

//==================== 时钟生成 ====================
initial begin
    clk = 0;
    forever #5 clk = ~clk;  // 100MHz
end

//==================== DUT 例化 ====================
axis_cnn_top dut (
    .clk(clk),
    .rst_n(rst_n),
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

//==================== 主测试流程 ====================
initial begin
    // 初始化
    rst_n = 0;
    s_axis_tdata  = 0;
    s_axis_tvalid = 0;
    s_axis_tlast  = 0;
    m_axis_tready = 1;
    conv_weight_wr_en = 0;
    conv_weight_out_ch = 0;
    conv_weight_addr = 0;
    conv_weight_data = 0;
    conv_bias_wr_en = 0;
    conv_bias_out_ch = 0;
    conv_bias_data = 0;
    bn_alpha_wr_en = 0;
    bn_alpha_ch = 0;
    bn_alpha_data = 0;
    bn_beta_wr_en = 0;
    bn_beta_ch = 0;
    bn_beta_data = 0;
    fc_weight_wr_en = 0;
    fc_weight_out_idx = 0;
    fc_weight_in_idx = 0;
    fc_weight_data = 0;
    fc_bias_wr_en = 0;
    fc_bias_out_idx = 0;
    fc_bias_data = 0;
    input_idx  = 0;
    output_idx = 0;
    err_cnt    = 0;
    tlast_cnt  = 0;

    // 读取 hex 文件
    $readmemh("input.hex", input_mem);
    $readmemh("conv_weights.hex", conv_weight_mem);
    $readmemh("conv_biases.hex", conv_bias_mem);
    $readmemh("bn_alpha.hex", bn_alpha_mem);
    $readmemh("bn_beta.hex", bn_beta_mem);
    $readmemh("fc_weights.hex", fc_weight_mem);
    $readmemh("fc_biases.hex", fc_bias_mem);
    $readmemh("golden_output.hex", golden_mem);

    $display("============================================================");
    $display("Testbench Start (Conv + BN + Pool + ReLU + FC Cascade)");
    $display("  INPUT_CNT        = %0d (8x8x1)", INPUT_CNT);
    $display("  CONV_WEIGHT_CNT  = %0d (6x9)", CONV_WEIGHT_CNT);
    $display("  CONV_BIAS_CNT    = %0d", CONV_BIAS_CNT);
    $display("  BN_ALPHA_CNT     = %0d", BN_ALPHA_CNT);
    $display("  BN_BETA_CNT      = %0d", BN_BETA_CNT);
    $display("  FC_WEIGHT_CNT    = %0d (10x54)", FC_WEIGHT_CNT);
    $display("  FC_BIAS_CNT      = %0d", FC_BIAS_CNT);
    $display("  OUTPUT_CNT       = %0d", OUTPUT_CNT);
    $display("============================================================");

    // 复位
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    //==================== 配置卷积权重 ====================
    $display("[INFO] Configuring conv weights...");
    for (i = 0; i < CONV_WEIGHT_CNT; i = i + 1) begin
        @(negedge clk);
        conv_weight_wr_en    = 1;
        conv_weight_out_ch  = i / 9;
        conv_weight_addr   = i % 9;
        conv_weight_data     = conv_weight_mem[i];
    end
    @(negedge clk);
    conv_weight_wr_en = 0;

    //==================== 配置卷积偏置 ====================
    $display("[INFO] Configuring conv biases...");
    for (i = 0; i < CONV_BIAS_CNT; i = i + 1) begin
        @(negedge clk);
        conv_bias_wr_en   = 1;
        conv_bias_out_ch = i;
        conv_bias_data    = conv_bias_mem[i];
    end
    @(negedge clk);
    conv_bias_wr_en = 0;

    //==================== 配置 BN alpha ====================
    $display("[INFO] Configuring BN alpha...");
    for (i = 0; i < BN_ALPHA_CNT; i = i + 1) begin
        @(negedge clk);
        bn_alpha_wr_en = 1;
        bn_alpha_ch    = i;
        bn_alpha_data  = bn_alpha_mem[i];
    end
    @(negedge clk);
    bn_alpha_wr_en = 0;

    //==================== 配置 BN beta ====================
    $display("[INFO] Configuring BN beta...");
    for (i = 0; i < BN_BETA_CNT; i = i + 1) begin
        @(negedge clk);
        bn_beta_wr_en = 1;
        bn_beta_ch    = i;
        bn_beta_data  = bn_beta_mem[i];
    end
    @(negedge clk);
    bn_beta_wr_en = 0;

    //==================== 配置FC权重 ====================
    $display("[INFO] Configuring FC weights...");
    for (i = 0; i < FC_WEIGHT_CNT; i = i + 1) begin
        @(negedge clk);
        fc_weight_wr_en    = 1;
        fc_weight_out_idx  = i / 54;
        fc_weight_in_idx   = i % 54;
        fc_weight_data     = fc_weight_mem[i];
    end
    @(negedge clk);
    fc_weight_wr_en = 0;

    //==================== 配置FC偏置 ====================
    $display("[INFO] Configuring FC biases...");
    for (i = 0; i < FC_BIAS_CNT; i = i + 1) begin
        @(negedge clk);
        fc_bias_wr_en   = 1;
        fc_bias_out_idx = i;
        fc_bias_data    = fc_bias_mem[i];
    end
    @(negedge clk);
    fc_bias_wr_en = 0;

    repeat (5) @(posedge clk);

    //==================== 启动输入驱动和输出收集 ====================
    fork
        // 输入驱动线程
        begin
            $display("[INFO] Driving input stream...");
            for (input_idx = 0; input_idx < INPUT_CNT; input_idx = input_idx + 1) begin
                @(negedge clk);
                s_axis_tdata  = input_mem[input_idx];
                s_axis_tvalid = 1;
                s_axis_tlast  = (input_idx == INPUT_CNT - 1) ? 1 : 0;
                // 卷积模块第一个数据特殊处理
                if (input_idx == 0) @(posedge clk);
                while (!s_axis_tready) @(posedge clk);
            end
            @(negedge clk);
            s_axis_tvalid = 0;
            s_axis_tlast  = 0;
            $display("[INFO] Input stream done.");
        end

        // 输出收集线程
        begin
            while (output_idx < OUTPUT_CNT) begin
                @(negedge clk);
                if (m_axis_tvalid && m_axis_tready) begin
                    output_mem[output_idx] = m_axis_tdata;
                    if (m_axis_tlast) begin
                        tlast_cnt = tlast_cnt + 1;
                        $display("[INFO] Output TLAST at idx=%0d", output_idx);
                    end
                    output_idx = output_idx + 1;
                end
            end
            $display("[INFO] Output stream done.");
        end
    join

    $display("[INFO] Output TLAST count=%0d", tlast_cnt);

    // 等待几个周期
    repeat (10) @(posedge clk);

    //==================== 比对结果 ====================
    $display("");
    $display("============================================================");
    $display("Comparison Results");
    $display("============================================================");
    for (i = 0; i < OUTPUT_CNT; i = i + 1) begin
        if (output_mem[i] !== golden_mem[i]) begin
            $display("[ERROR] idx=%0d: expected=%0d, got=%0d",
                     i, golden_mem[i], output_mem[i]);
            err_cnt = err_cnt + 1;
        end
    end

    if (err_cnt == 0) begin
        $display("");
        $display("[PASS] All %0d outputs match golden vector!", OUTPUT_CNT);
        $display("");
        $display("All outputs:");
        for (i = 0; i < OUTPUT_CNT; i = i + 1) begin
            $display("  idx=%0d: golden=%0d, rtl=%0d", i, golden_mem[i], output_mem[i]);
        end
    end else begin
        $display("");
        $display("[FAIL] %0d / %0d outputs mismatched!", err_cnt, OUTPUT_CNT);
    end

    $display("============================================================");
    $finish;
end

//==================== 超时保护 ====================
initial begin
    #200000;  // 200us 超时
    $display("[TIMEOUT] Simulation timed out!");
    $finish;
end

endmodule
