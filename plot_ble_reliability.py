# import numpy as np
# import matplotlib.pyplot as plt


# # =========================
# # 1. 基础函数
# # =========================
# def p_no_collision(C_eff, N):
#     """
#     单次尝试中，victim 选到未碰撞信道的概率
#     C_eff: 有效跳频信道数
#     N: 活跃连接数（包含 victim 自身）
#     """
#     C_eff = np.asarray(C_eff, dtype=float)
#     N = np.asarray(N, dtype=float)
#     return np.power(1.0 - 1.0 / C_eff, N - 1.0)


# def p_phy_success(BER, L):
#     """
#     未碰撞时，L bit packet 的物理层成功概率
#     """
#     BER = np.asarray(BER, dtype=float)
#     L = np.asarray(L, dtype=float)
#     return np.power(1.0 - BER, L)


# def p_single_try_success(C_eff, N, BER, L):
#     """
#     单次尝试成功概率：
#     p_s = P(no collision) * P(phy success | no collision)
#     """
#     return p_no_collision(C_eff, N) * p_phy_success(BER, L)


# def reliability_simple(C_eff, N, BER, L, r):
#     """
#     简化模型：
#     一个 connection event 内有 r 次等效独立尝试机会
#     R = 1 - (1 - p_s)^r
#     """
#     p_s = p_single_try_success(C_eff, N, BER, L)
#     return 1.0 - np.power(1.0 - p_s, r)


# # =========================
# # 2. 引入 fail(open) / fail(close)
# # =========================
# def reliability_open_close(C_eff, N, BER, L, x, eta):
#     """
#     event-level 模型：
#     - 单次成功概率: p_s
#     - 单次失败概率: p_f = 1 - p_s
#     - fail(open): p_fo = (1 - eta) * p_f
#     - fail(close): p_fc = eta * p_f

#     一个 connection event 内最多允许 x 次 transaction。
#     只有 fail(open) 才能继续留在当前 event 里重试。
#     fail(close) 会立刻结束当前 event。

#     最终：
#     R_event = sum_{k=1..x} p_s * p_fo^(k-1)
#             = p_s * (1 - p_fo^x) / (1 - p_fo)
#     """
#     p_s = p_single_try_success(C_eff, N, BER, L)
#     p_f = 1.0 - p_s
#     p_fo = (1.0 - eta) * p_f

#     # 数值稳定处理
#     denom = 1.0 - p_fo
#     R = np.where(
#         np.abs(denom) < 1e-12,
#         x * p_s,
#         p_s * (1.0 - np.power(p_fo, x)) / denom
#     )

#     # 裁剪到 [0, 1]
#     return np.clip(R, 0.0, 1.0)


# # =========================
# # 3. 从非均匀信道分布计算 C_eff
# # =========================
# def effective_channel_number(f):
#     """
#     由信道选择概率分布 f(x) 计算等效信道数:
#     C_eff = 1 / sum_x f(x)^2

#     f: 一维数组，表示各信道被选中的概率，和应为 1
#     """
#     f = np.asarray(f, dtype=float)
#     f = f / np.sum(f)
#     return 1.0 / np.sum(f ** 2)


# # =========================
# # 4. 画图示例 1：R vs N
# # =========================
# def plot_reliability_vs_N():
#     BER = 1e-4
#     L = 160          # 例如 20 Bytes payload 对应可自行换成更完整 bit 长
#     x = 4            # 一个 connection event 内最多 4 次 transaction 机会
#     eta = 0.35       # 失败中 35% 导致 fail(close)
#     N_values = np.arange(1, 41)

#     C_list = [5, 10, 20, 37]

#     plt.figure(figsize=(7, 5))
#     for C_eff in C_list:
#         R = reliability_open_close(C_eff=C_eff, N=N_values, BER=BER, L=L, x=x, eta=eta)
#         plt.plot(N_values, R, label=f"C_eff={C_eff}")

#     plt.xlabel("Number of active BLE connections N")
#     plt.ylabel("Packet reception reliability per event")
#     plt.title("Reliability vs N")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# # =========================
# # 5. 画图示例 2：R vs C_eff
# # =========================
# def plot_reliability_vs_C():
#     BER = 1e-4
#     L = 160
#     x = 4
#     eta = 0.35
#     C_values = np.arange(2, 38)

#     N_list = [2, 5, 10, 20]

#     plt.figure(figsize=(7, 5))
#     for N in N_list:
#         R = reliability_open_close(C_eff=C_values, N=N, BER=BER, L=L, x=x, eta=eta)
#         plt.plot(C_values, R, label=f"N={N}")

#     plt.xlabel("Effective number of hopping channels C_eff")
#     plt.ylabel("Packet reception reliability per event")
#     plt.title("Reliability vs C_eff")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# # =========================
# # 6. 画图示例 3：比较“无 open/close”与“有 open/close”
# # =========================
# def plot_compare_models():
#     BER = 1e-4
#     L = 160
#     N_values = np.arange(1, 41)
#     C_eff = 20

#     R_simple = reliability_simple(C_eff=C_eff, N=N_values, BER=BER, L=L, r=4)
#     R_oc = reliability_open_close(C_eff=C_eff, N=N_values, BER=BER, L=L, x=4, eta=0.35)

#     plt.figure(figsize=(7, 5))
#     plt.plot(N_values, R_simple, label="Simple retry model")
#     plt.plot(N_values, R_oc, label="Open/close event model")

#     plt.xlabel("Number of active BLE connections N")
#     plt.ylabel("Packet reception reliability per event")
#     plt.title("Model comparison")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# # =========================
# # 7. 主程序
# # =========================
# if __name__ == "__main__":
#     plot_reliability_vs_N()
#     plot_reliability_vs_C()
#     plot_compare_models()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path('/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_reliability_outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def p_no_collision(C_eff, N):
    C_eff = np.asarray(C_eff, dtype=float)
    N = np.asarray(N, dtype=float)
    return np.power(1.0 - 1.0 / C_eff, N - 1.0)


def p_phy_success(BER, L):
    BER = np.asarray(BER, dtype=float)
    L = np.asarray(L, dtype=float)
    return np.power(1.0 - BER, L)


def p_single_try_success(C_eff, N, BER, L):
    return p_no_collision(C_eff, N) * p_phy_success(BER, L)


def reliability_simple(C_eff, N, BER, L, r):
    p_s = p_single_try_success(C_eff, N, BER, L)
    return 1.0 - np.power(1.0 - p_s, r)


def reliability_open_close(C_eff, N, BER, L, x, eta):
    p_s = p_single_try_success(C_eff, N, BER, L)
    p_f = 1.0 - p_s
    p_fo = (1.0 - eta) * p_f
    denom = 1.0 - p_fo
    R = np.where(np.abs(denom) < 1e-12, x * p_s, p_s * (1.0 - np.power(p_fo, x)) / denom)
    return np.clip(R, 0.0, 1.0)


def effective_channel_number(f):
    f = np.asarray(f, dtype=float)
    f = f / np.sum(f)
    return 1.0 / np.sum(f ** 2)


def get_parameter_table():
    return [
        {"Symbol": "C_eff", "Meaning": "Effective number of hopping channels", "Value": "5, 10, 20, 37 (varied by figure)", "Unit": "channels"},
        {"Symbol": "N", "Meaning": "Number of active BLE connections, including the victim link", "Value": "1-40 or selected values {2, 5, 10, 20}", "Unit": "connections"},
        {"Symbol": "BER", "Meaning": "Bit error rate under no strong co-channel collision", "Value": "1e-4", "Unit": "-"},
        {"Symbol": "L", "Meaning": "Packet length", "Value": "160", "Unit": "bits"},
        {"Symbol": "x", "Meaning": "Maximum transaction opportunities within one connection event", "Value": "4", "Unit": "transactions/event"},
        {"Symbol": "r", "Meaning": "Equivalent independent retry opportunities in the simple retry model", "Value": "4", "Unit": "attempts/event"},
        {"Symbol": "eta", "Meaning": "Fraction of failures that cause fail(close)", "Value": "0.35", "Unit": "-"},
    ]


def print_parameter_table():
    rows = get_parameter_table()
    header = ["Symbol", "Meaning", "Value", "Unit"]
    col_widths = {key: max(len(key), max(len(str(row[key])) for row in rows)) for key in header}
    line = " | ".join(key.ljust(col_widths[key]) for key in header)
    sep = "-+-".join("-" * col_widths[key] for key in header)
    print("\nPaper-style parameter table")
    print(sep)
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(str(row[key]).ljust(col_widths[key]) for key in header))
    print(sep)
    csv_path = OUTPUT_DIR / 'parameter_table.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(row[key]) for key in header) + '\n')
    return csv_path


def add_caption(fig, caption):
    fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=10)


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches='tight')
    return path


def plot_reliability_vs_N():
    BER = 1e-4
    L = 160
    x = 4
    eta = 0.35
    N_values = np.arange(1, 41)
    C_list = [5, 10, 20, 37]
    fig = plt.figure(figsize=(7, 5))
    for C_eff in C_list:
        R = reliability_open_close(C_eff=C_eff, N=N_values, BER=BER, L=L, x=x, eta=eta)
        plt.plot(N_values, R, label=f"$C_\\mathrm{{eff}}={C_eff}$")
    plt.xlabel("Number of active BLE connections, $N$")
    plt.ylabel("Packet reception reliability per event")
    plt.title("Reliability versus active BLE connection number")
    plt.grid(True)
    plt.legend()
    add_caption(fig, "Fig. 1. Event-level packet reception reliability versus the number of active BLE connections under different effective hopping-channel counts. Parameters: BER = 1e-4, L = 160 bits, x = 4, eta = 0.35.")
    path = save_figure(fig, 'fig1_reliability_vs_N.png')
    plt.close(fig)
    return path


def plot_reliability_vs_C():
    BER = 1e-4
    L = 160
    x = 4
    eta = 0.35
    C_values = np.arange(2, 38)
    N_list = [2, 5, 10, 20]
    fig = plt.figure(figsize=(7, 5))
    for N in N_list:
        R = reliability_open_close(C_eff=C_values, N=N, BER=BER, L=L, x=x, eta=eta)
        plt.plot(C_values, R, label=f"$N={N}$")
    plt.xlabel("Effective number of hopping channels, $C_\\mathrm{eff}$")
    plt.ylabel("Packet reception reliability per event")
    plt.title("Reliability versus effective channel number")
    plt.grid(True)
    plt.legend()
    add_caption(fig, "Fig. 2. Event-level packet reception reliability versus the effective number of hopping channels for different network loads. Parameters: BER = 1e-4, L = 160 bits, x = 4, eta = 0.35.")
    path = save_figure(fig, 'fig2_reliability_vs_Ceff.png')
    plt.close(fig)
    return path


def plot_compare_models():
    BER = 1e-4
    L = 160
    N_values = np.arange(1, 41)
    C_eff = 20
    R_simple = reliability_simple(C_eff=C_eff, N=N_values, BER=BER, L=L, r=4)
    R_oc = reliability_open_close(C_eff=C_eff, N=N_values, BER=BER, L=L, x=4, eta=0.35)
    fig = plt.figure(figsize=(7, 5))
    plt.plot(N_values, R_simple, label='Simple retry model')
    plt.plot(N_values, R_oc, label='Open/close event model')
    plt.xlabel("Number of active BLE connections, $N$")
    plt.ylabel("Packet reception reliability per event")
    plt.title("Comparison of retry abstractions")
    plt.grid(True)
    plt.legend()
    add_caption(fig, "Fig. 3. Comparison between the independent-retry abstraction and the fail(open)/fail(close)-aware event model. Parameters: C_eff = 20, BER = 1e-4, L = 160 bits, r = 4, x = 4, eta = 0.35.")
    path = save_figure(fig, 'fig3_model_comparison.png')
    plt.close(fig)
    return path


def plot_reliability_vs_BER():
    C_eff = 20
    L = 160
    x = 4
    eta = 0.35
    BER_values = np.logspace(-6, -2, 200)
    N_list = [2, 5, 10, 20]
    fig = plt.figure(figsize=(7, 5))
    for N in N_list:
        R = reliability_open_close(C_eff=C_eff, N=N, BER=BER_values, L=L, x=x, eta=eta)
        plt.semilogx(BER_values, R, label=f"$N={N}$")
    plt.xlabel('Bit error rate, BER')
    plt.ylabel('Packet reception reliability per event')
    plt.title('Reliability versus bit error rate')
    plt.grid(True)
    plt.legend()
    add_caption(fig, "Fig. 4. Event-level packet reception reliability versus bit error rate for different numbers of active BLE connections. Parameters: C_eff = 20, L = 160 bits, x = 4, eta = 0.35.")
    path = save_figure(fig, 'fig4_reliability_vs_BER.png')
    plt.close(fig)
    return path


def main():
    print('Generating BLE reliability figures and parameter table...')
    csv_path = print_parameter_table()
    figure_paths = [plot_reliability_vs_N(), plot_reliability_vs_C(), plot_compare_models(), plot_reliability_vs_BER()]
    print('\nSaved files:')
    print(f'- {csv_path}')
    for path in figure_paths:
        print(f'- {path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
