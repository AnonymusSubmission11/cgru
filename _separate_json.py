import os, json, random

def save_jsonl(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    # --- Config (edit here) ---
    input_path  = "ddpo_pytorch/assets/CoPro_v1.0.json"  # expects the exact format with "ID_train_data"
    out_dir     = "ddpo_pytorch/assets/CoPro"
    seed        = 1234
    critic_pct  = 0.20   # 20%
    policy_pct  = 0.40   # 40%
    filter_category = "sexual"  # Only load items with this category

    # --- Load ---
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["ID_train_data"]  # minimal: expect this field
    assert isinstance(items, list), "ID_train_data must be a list"

    # --- Filter by category ---
    items = [item for item in items if item.get("category", "").lower() == filter_category.lower()]
    print(f"Filtered to {len(items)} items with category='{filter_category}'")

    if len(items) == 0:
        print(f"⚠️  No items found with category='{filter_category}'. Exiting.")
        return

    n = len(items)
    n_critic = max(1, round(n * critic_pct))
    n_policy = max(1, round(n * policy_pct))
    print(f"Total: {n} | critic 5%: {n_critic} | policy 10%: {n_policy}")

    # --- Disjoint random splits ---
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    critic_idxs = set(idxs[:n_critic])
    policy_idxs = set(idxs[n_critic:n_critic + n_policy])

    # --- Save ---
    critic_items = [items[i] for i in sorted(critic_idxs)]
    policy_items = [items[i] for i in sorted(policy_idxs)]

    os.makedirs(out_dir, exist_ok=True)
    save_jsonl(critic_items, os.path.join(out_dir, "cgru_critic_20_sexual.jsonl"))
    save_jsonl(policy_items, os.path.join(out_dir, "ddpo_policy_40_sexual.jsonl"))

    print(f"✅ Saved:")
    print(f" - {os.path.join(out_dir, 'cgru_critic_20_sexual.jsonl')}  ({len(critic_items)} items)")
    print(f" - {os.path.join(out_dir, 'ddpo_policy_40_sexual.jsonl')} ({len(policy_items)} items)")

if __name__ == "__main__":
    main()
