import torch

from normalizer_python import TextNormalizer
from src.setup import fetch_stop_words, PATH_TO_STOP_WORDS
from src.model.encoder import PaperEncoder
from src.rag.engine import RAGEngine
from src.db import connection
from src.embed import WEIGHTS_PATH


def _clear_terminal() -> None:
    """Clears the terminal screen using ANSI escape sequences."""
    print("\033[2J\033[H", end="", flush=True)


def _parse_k(cmd_parts: list[str]) -> int:
    """Parses the 'k' integer value from a command string. Defaults to 5 if invalid."""
    try:
        if len(cmd_parts) > 1:
            k_val = int(cmd_parts[1])
            if k_val > 0:
                return k_val
            else:
                print("Value for k must be greater than 0. Keeping previous k.")
    except ValueError:
        print(f"Invalid number format: '{cmd_parts[1]}'. Usage: /k <integer>")
    
    return 5


def _display_sources(last_papers: list[dict]) -> None:
    """Prints the metadata of the papers retrieved in the most recent query."""
    if not last_papers:
        print("No sources to display. Please run a RAG query first.")
        return
    
    print("\n--- Sources from Last Query ---")
    for idx, paper in enumerate(last_papers, start=1):
        title = paper.get("title", "N/A").strip()
        paper_id = paper.get("paper_id", "N/A")
        sim = paper.get("cosine_similarity", 0.0)
        date = paper.get("published_date", "Unknown")
        print(f"[{idx}] {title}")
        print(f"    ID: {paper_id} | Date: {date} | Similarity: {sim:.4f}")
    print("-------------------------------\n")


def _display_help() -> None:
    """Displays the CLI available commands."""
    print("\nAvailable Commands:")
    print("  /help          Show this help menu")
    print("  /clear         Clear the terminal screen")
    print("  /k <number>    Set the number of retrieved context documents (e.g., /k 3)")
    print("  /sources       Show details of the papers from the last search")
    print("  /exit, /quit   Close the application\n")


def main():
    # init text normalizer
    stop_words: set[str] = fetch_stop_words(path=PATH_TO_STOP_WORDS)
    normalizer = TextNormalizer(stop_words)

    # detect avialable device 
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = checkpoint["projection_state_dict"]

    # extract output projection dimension dynamically from weight tensor shape [out_features, in_features]
    projection_dim = state_dict["weight"].shape[0]

    # instantiate model with detected dimension and apply weights
    model = PaperEncoder(projection_dim=projection_dim)
    model._projection.load_state_dict(state_dict)
    model.eval()

    # init conn pool
    connection.init_pool()

    rag_engine = RAGEngine(
        pool=connection.pool,
        normalizer=normalizer,
        encoder=model,
        model_name="gemini-3.6-flash"
    )

    current_k = 5
    last_papers = []

    try:
        while True:
            try:
                # capture user input 
                user_input = input("\n[You] > ").strip()

                if not user_input:
                    continue

                # command Routing
                if user_input.startswith("/"):
                    cmd_parts = user_input.split()
                    command = cmd_parts[0].lower()

                    if command in ("/exit", "/quit"):
                        print("Exiting...")
                        break

                    elif command == "/clear":
                        _clear_terminal()

                    elif command == "/k":
                        # update k dynamically from args
                        current_k = _parse_k(cmd_parts)
                        print(f"Retrieval size updated to k={current_k}")

                    elif command == "/sources":
                        # print full details from `last_papers`
                        _display_sources(last_papers)

                    elif command == "/help":
                        _display_help()

                    else:
                        print("Unknown command. Type /help for options.")

                    continue

                # RAG Query Handling
                print("Thinking... [Generating embedding and querying vector DB]")
                answer, last_papers = rag_engine.query(user_input, k=current_k)

                # print response
                print(f"\n[Assistant]:\n{answer}")

            except KeyboardInterrupt:
                # user pressed Ctrl+C inside the loop -> clear current line and keep going
                print("\nQuery interrupted. Type /exit to close.")
            except Exception as e:
                print(f"An error occurred: {e}")

    except EOFError:
        # user pressed Ctrl+D -> graceful exit
        pass

    finally:
        # teardown
        print("Closing database connections...")
        connection.close_pool()
        print("Goodbye!")

if __name__ == "__main__":
    main()
