from embed import Config, DocumentIndexer, Path

def main():
    config = Config()
    indexer = DocumentIndexer(config)
    results = indexer.search("where am i")
    print("Search Results:")
    print(results)

if __name__ == "__main__":
    main()