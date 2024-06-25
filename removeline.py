import os

def process_text_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            processed_lines = []
            for i in range(len(lines)):
                if lines[i].strip() == "":
                    processed_lines.append(lines[i])
                else:
                    if i + 1 < len(lines) and lines[i + 1].strip() == "":
                        processed_lines.append(lines[i])
                    else:
                        processed_lines.append(lines[i].rstrip() + " ")

            # Join the processed lines into a single string
            processed_content = "".join(processed_lines)

            # Write the processed content back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(processed_content)

# Replace 'your_folder_path' with the actual path to your folder containing the text files
folder_path = 'books'
process_text_files(folder_path)
