import os
import random

def save_txt(lines, path):
    """Save lines to a text file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")

def main():
    # --- Config (edit here) ---
    input_path  = "ddpo_pytorch/assets/nsfw/nsfw-dataset-llava.txt"
    out_dir     = "ddpo_pytorch/assets/nsfw"
    seed        = 1234
    critic_pct  = 0.10   # 10%
    policy_pct  = 0.20   # 10%

    # --- Load ---
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    n = len(lines)
    n_critic = max(1, round(n * critic_pct))
    n_policy = max(1, round(n * policy_pct))
    print(f"Total: {n} | critic 10%: {n_critic} | policy 20%: {n_policy}")

    # --- Disjoint random splits ---
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    critic_idxs = set(idxs[:n_critic])
    policy_idxs = set(idxs[n_critic:n_critic + n_policy])

    # --- Save ---
    critic_lines = [lines[i] for i in sorted(critic_idxs)]
    policy_lines = [lines[i] for i in sorted(policy_idxs)]

    os.makedirs(out_dir, exist_ok=True)
    save_txt(critic_lines, os.path.join(out_dir, "nsfw_critic_10.txt"))
    save_txt(policy_lines, os.path.join(out_dir, "nsfw_policy_20.txt"))

    print(f"✅ Saved:")
    print(f" - {os.path.join(out_dir, 'nsfw_critic_10.txt')}  ({len(critic_lines)} lines)")
    print(f" - {os.path.join(out_dir, 'nsfw_policy_20.txt')} ({len(policy_lines)} lines)")

if __name__ == "__main__":
    main()