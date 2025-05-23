from graphviz import Digraph

def draw_ACBiFPN():
    dot = Digraph("ACBiFPN", format="png")

    # 输入特征图
    dot.node("X1", "Input Feature 1", shape="rect", style="filled", fillcolor="lightblue")
    dot.node("X2", "Input Feature 2", shape="rect", style="filled", fillcolor="lightblue")
    dot.node("X3", "Input Feature 3", shape="rect", style="filled", fillcolor="lightblue")

    # 权重参数
    dot.node("W", "Learnable Weights (w)", shape="ellipse", style="filled", fillcolor="lightgray")

    # 归一化
    dot.node("Norm", "Weight Normalization", shape="parallelogram", style="filled", fillcolor="yellow")

    # 加权特征
    dot.node("Weighted1", "X1 * w1", shape="rect", style="filled", fillcolor="lightgreen")
    dot.node("Weighted2", "X2 * w2", shape="rect", style="filled", fillcolor="lightgreen")
    dot.node("Weighted3", "X3 * w3", shape="rect", style="filled", fillcolor="lightgreen")

    # 特征拼接
    dot.node("Concat", "Concatenation", shape="parallelogram", style="filled", fillcolor="orange")

    # 通道压缩
    dot.node("Conv", "1x1 Conv (Channel Compression)", shape="rect", style="filled", fillcolor="red")

    # 最终输出
    dot.node("Output", "Fused Output", shape="rect", style="filled", fillcolor="lightblue")

    # 连接权重
    dot.edge("W", "Norm")
    dot.edge("Norm", "Weighted1")
    dot.edge("Norm", "Weighted2")
    dot.edge("Norm", "Weighted3")

    # 连接输入特征图
    dot.edge("X1", "Weighted1")
    dot.edge("X2", "Weighted2")
    dot.edge("X3", "Weighted3")

    # 连接加权特征
    dot.edge("Weighted1", "Concat")
    dot.edge("Weighted2", "Concat")
    dot.edge("Weighted3", "Concat")

    # 连接拼接到压缩
    dot.edge("Concat", "Conv")

    # 连接最终输出
    dot.edge("Conv", "Output")

    # 渲染保存图像
    dot.render("ACBiFPN_structure", view=True)

# 生成并展示 ACBiFPN 结构图
draw_ACBiFPN()
