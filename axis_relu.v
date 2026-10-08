//=============================================================================
// ReLU 激活函数加速器 (axis_relu)
// AXI-Stream 接口，逐元素计算 max(0, x)
// 输入/输出均为有符号数，一级寄存器打拍
//=============================================================================
module axis_relu #(
    parameter DATA_W = 8     // 数据位宽（有符号）
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //==================== 输入 AXI-Stream ====================
    input  wire signed [DATA_W-1:0]  s_axis_tdata,
    input  wire                       s_axis_tvalid,
    output wire                       s_axis_tready,
    input  wire                       s_axis_tlast,

    //==================== 输出 AXI-Stream ====================
    output reg signed [DATA_W-1:0]  m_axis_tdata,
    output reg                       m_axis_tvalid,
    input  wire                       m_axis_tready,
    output reg                       m_axis_tlast
);

//=============================================================================
// 组合逻辑：ReLU 计算 max(0, x)
//=============================================================================
wire signed [DATA_W-1:0] relu_out = ($signed(s_axis_tdata) > 0) ?
                                       s_axis_tdata : {DATA_W{1'b0}};

//=============================================================================
// 一级寄存器打拍（Testbench m_axis_tready 恒为 1，无背压）
//=============================================================================
assign s_axis_tready = 1'b1;   // 一直就绪

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        m_axis_tvalid <= 1'b0;
        m_axis_tdata  <= {DATA_W{1'b0}};
        m_axis_tlast  <= 1'b0;
    end else begin
        m_axis_tvalid <= s_axis_tvalid;
        m_axis_tdata  <= relu_out;
        m_axis_tlast  <= s_axis_tlast;
    end
end

endmodule
