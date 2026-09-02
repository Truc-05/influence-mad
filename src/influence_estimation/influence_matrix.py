import numpy as np

AGENT_NAMES = ["careful", "skeptic", "bigpicture"]


def _norm(text):
    return (text or "").strip().lower()


def get_agent_names(debate_logs):
    """Lấy danh sách agent theo thứ tự ổn định từ log (final_answers)."""
    for log in debate_logs:
        fa = log.get("final_answers")
        if fa:
            return list(fa.keys())
    return list(AGENT_NAMES)


def build_influence_matrix(debate_logs, agent_names=None):
    """Influence Matrix N x N, đo ĐỦ 2 CHIỀU cho mọi cặp peer.

    matrix[i][j] = cường độ agent i ẢNH HƯỞNG lên agent j
                 = P(j chuyển sang đáp án của i | i đưa ra đáp án khác j).

    Cách đếm cho mỗi vòng r >= 1, với answers vòng r-1 (prev) và vòng r (curr):
      - Cơ hội i tác động j (totals[i][j]) khi i_prev != j_prev
        (i đang chào một lựa chọn khác với j).
      - Ghi nhận ảnh hưởng (counts[i][j]) khi thêm điều kiện:
        j_curr == i_prev  và  j_curr != j_prev  (j đã dịch sang đúng đáp án i).
    """
    if agent_names is None:
        agent_names = get_agent_names(debate_logs)
    n = len(agent_names)
    idx = {name: k for k, name in enumerate(agent_names)}

    counts = np.zeros((n, n))
    totals = np.zeros((n, n))

    for log in debate_logs:
        rounds = log["rounds"]
        for r in range(1, len(rounds)):
            prev = rounds[r - 1]["answers"]
            curr = rounds[r]["answers"]
            for i_name in agent_names:
                for j_name in agent_names:
                    if i_name == j_name:
                        continue
                    if i_name not in prev or j_name not in prev or j_name not in curr:
                        continue
                    i_prev = _norm(prev[i_name]["answer"])
                    j_prev = _norm(prev[j_name]["answer"])
                    j_curr = _norm(curr[j_name]["answer"])
                    if i_prev == j_prev:
                        continue 
                    i, j = idx[i_name], idx[j_name]
                    totals[i][j] += 1
                    if j_curr == i_prev and j_curr != j_prev:
                        counts[i][j] += 1

    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if totals[i][j] > 0:
                matrix[i][j] = counts[i][j] / totals[i][j]
    return matrix
