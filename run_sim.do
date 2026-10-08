#=============================================================================
# axis_cnn_top ModelSim 自动化仿真脚本
# 编译顺序：Conv -> BN -> Pool -> ReLU -> FC -> Top -> TB
# 用法：在 ModelSim 控制台执行 do run_sim.do
#=============================================================================

#==================== 目录设置 ====================
set project_root "F:/20260822simulation/axis_cnn_top"
set conv_dir     "F:/20260822simulation/axis_conv3x3_mc"
set bn_dir       "F:/20260822simulation/axis_bn"
set pool_dir     "F:/20260822simulation/axis_maxpool2x2"
set relu_dir     "F:/20260822simulation/axis_relu"
set fc_dir       "F:/20260822simulation/axis_fc"

#==================== [1/5] 清理旧编译结果 ====================
echo "\[1/5\] 清理旧编译结果..."
file delete -force work/_opt
if {[file exists work]} {
    vdel -all -lib work
}
vlib work
vmap work work

#==================== [2/5] 编译所有模块 ====================
echo "\[2/5\] 编译RTL + Testbench..."

# 子模块
vlog -reportprogress 300 -work work $conv_dir/axis_conv3x3_mc.v
vlog -reportprogress 300 -work work $bn_dir/axis_bn.v
vlog -reportprogress 300 -work work $pool_dir/axis_maxpool2x2.v
vlog -reportprogress 300 -work work $relu_dir/axis_relu.v
vlog -reportprogress 300 -work work $fc_dir/axis_fc.v

# 顶层 + Testbench
vlog -reportprogress 300 -work work $project_root/axis_cnn_top.v
vlog -reportprogress 300 -work work $project_root/tb_axis_cnn_top.v

#==================== [3/5] 启动仿真 ====================
echo "\[3/5\] 启动仿真 tb_axis_cnn_top"
vsim -voptargs="+acc" work.tb_axis_cnn_top

#==================== [4/5] 加载波形信号 ====================
echo "\[4/5\] 加载波形信号..."
add wave -divider "Input"
add wave -radix decimal /tb_axis_cnn_top/s_axis_tdata
add wave /tb_axis_cnn_top/s_axis_tvalid
add wave /tb_axis_cnn_top/s_axis_tready
add wave /tb_axis_cnn_top/s_axis_tlast

add wave -divider "Output"
add wave -radix decimal /tb_axis_cnn_top/m_axis_tdata
add wave /tb_axis_cnn_top/m_axis_tvalid
add wave /tb_axis_cnn_top/m_axis_tready
add wave /tb_axis_cnn_top/m_axis_tlast

add wave -divider "Conv_to_BN"
add wave -radix decimal /tb_axis_cnn_top/dut/conv_to_bn_tdata
add wave /tb_axis_cnn_top/dut/conv_to_bn_tvalid
add wave /tb_axis_cnn_top/dut/conv_to_bn_tready

add wave -divider "BN_to_Pool"
add wave -radix decimal /tb_axis_cnn_top/dut/bn_to_pool_tdata
add wave /tb_axis_cnn_top/dut/bn_to_pool_tvalid
add wave /tb_axis_cnn_top/dut/bn_to_pool_tready

add wave -divider "Pool_to_ReLU"
add wave -radix decimal /tb_axis_cnn_top/dut/pool_to_relu_tdata
add wave /tb_axis_cnn_top/dut/pool_to_relu_tvalid
add wave /tb_axis_cnn_top/dut/pool_to_relu_tready

add wave -divider "ReLU_to_FC"
add wave -radix decimal /tb_axis_cnn_top/dut/relu_to_fc_tdata
add wave /tb_axis_cnn_top/dut/relu_to_fc_tvalid
add wave /tb_axis_cnn_top/dut/relu_to_fc_tready

view wave
wave zoom full

#==================== [5/5] 运行仿真 ====================
echo "\[5/5\] 开始运行仿真 run -all"
run -all

echo "=================================================="
echo "仿真执行结束，请查看Transcript打印PASS/FAIL信息"
echo "=================================================="
