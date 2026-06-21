from pathlib import Path
import json


def main() -> None:
    base_dir = Path("data/simulations")
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return

    for path in sorted(base_dir.iterdir(), key=lambda p: p.name):
        if not path.is_dir():
            continue
        for json_path in sorted(path.glob("*.json"), key=lambda p: p.name):
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            p = "../tau2-new" + str(json_path.absolute()).split("/tau2-new")[1]
            if "path" in data and data["path"] == p:
                continue  # Skip if path is already correct
            data["path"] = p
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")


if __name__ == "__main__":
    main()
