import os
import sys

def copy_to_clipboard(text):
    try:
        import pyperclip
        pyperclip.copy(text)
        print("Core prompt copied to clipboard.")
        return True
    except ImportError:
        # Fallback to OS-specific clipboard utilities
        if sys.platform == "darwin":
            os.system(f"echo '{text}' | pbcopy")
            print("Core prompt copied to clipboard (macOS pbcopy).")
            return True
        elif sys.platform == "win32":
            os.system(f"echo {text.strip()} | clip")
            print("Core prompt copied to clipboard (Windows clip).")
            return True
        elif sys.platform.startswith("linux"):
            os.system(f"echo '{text}' | xclip -selection clipboard")
            print("Core prompt copied to clipboard (xclip).")
            return True
        else:
            print("Clipboard copy not supported on this platform.")
            return False

def main(args):
    prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "cline_docs", "prompts", "core_prompt(put this in Custom Instructions).md"
    )
    if not os.path.exists(prompt_path):
        print(f"Core prompt file not found: {prompt_path}")
        return
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    copied = copy_to_clipboard(prompt)
    print("\n=== Cline Extension Configuration ===")
    print("1. Open VS Code settings for the Cline extension.")
    print("2. Paste the copied core prompt into the 'Custom Instructions' or system prompt field.")
    print("3. Save settings and reload the extension if needed.")
    if not copied:
        print("\nCould not copy to clipboard automatically. Please copy the following prompt manually:\n")
        print(prompt)
