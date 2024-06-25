import os
from tqdm import tqdm
from collections import Counter

# Path to the directory containing normalized books
normalized_books_dir = "books_categorized_2000_normalized"
output_file_prefix = "merged_sentences"


# Function to merge all unique sentences into multiple files with a maximum of 500,000 lines each
def merge_sentences(input_dir, output_file_prefix, max_lines_per_file=100000):
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    unique_sentences = set()
    word_counter = Counter()
    total_sentences = 0
    current_line_count = 0
    file_index = 1

    output_file = f"{output_file_prefix}_{file_index}.txt"
    outfile = open(output_file, "w", encoding="utf-8")

    for file in tqdm(files, desc="Merging Sentences"):
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if len(line) > 40:
                    sentence_key = (line[:20], line[-20:])
                else:
                    sentence_key = (line[:20],) if len(line) >= 20 else (line[-20:],)
                if sentence_key not in unique_sentences:
                    unique_sentences.add(sentence_key)
                    outfile.write(line + "\n")
                    total_sentences += 1
                    current_line_count += 1
                    words = line.split()
                    word_counter.update(words)

                    if current_line_count >= max_lines_per_file:
                        outfile.close()
                        file_index += 1
                        output_file = f"{output_file_prefix}_{file_index}.txt"
                        outfile = open(output_file, "w", encoding="utf-8")
                        current_line_count = 0

    outfile.close()
    return total_sentences, len(word_counter)


# Main function to run the merging process
def main():
    total_sentences, unique_words_count = merge_sentences(
        normalized_books_dir, output_file_prefix
    )
    print(
        f"All unique sentences have been merged into files with prefix '{output_file_prefix}'"
    )
    print(f"Total number of sentences: {total_sentences}")
    print(f"Total number of unique words: {unique_words_count}")


if __name__ == "__main__":
    main()
