import networkx as nx


def compute_influence_scores(influence_matrix, agent_names, damping_factor=0.85):
    """Tính Influence Score cho từng agent bằng PageRank.

    influence_matrix[i][j] = mức độ agent i ảnh hưởng lên agent j.

    PageRank chấm điểm cao cho node ĐƯỢC nhiều node khác trỏ tới. Một
    "influencer" quan trọng khi nhiều agent bị nó tác động -> cạnh phải đi từ
    người bị ảnh hưởng (j) TỚI người ảnh hưởng (i), giống như trang web quan
    trọng khi được nhiều trang khác link tới.

    (Trước đây cạnh thêm theo chiều i -> j nên PageRank lại nâng điểm cho
    agent BỊ ảnh hưởng thay vì agent gây ảnh hưởng -> đảo ngược thứ hạng.)
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(agent_names)

    n = len(agent_names)
    for i in range(n):
        for j in range(n):
            weight = influence_matrix[i][j]
            if weight > 0:
                # cạnh: influenced (j) -> influencer (i)
                graph.add_edge(agent_names[j], agent_names[i], weight=weight)

    scores = nx.pagerank(graph, alpha=damping_factor, weight="weight")
    return scores
