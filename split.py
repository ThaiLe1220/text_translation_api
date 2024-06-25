import re

def split_text(text, max_length):
    # Split text by sentences based on delimiters
    sentences = re.split(r'(?<=\.|,|;|"|!|\?)\s+', text)
    
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        
        # Check if adding this sentence would exceed the max length
        if current_length + sentence_length + 1 > max_length:  # +1 for space or delimiter
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = sentence_length
        else:
            current_chunk += (" " + sentence) if current_chunk else sentence
            current_length += sentence_length + 1  # +1 for space or delimiter

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Example usage
text = """The whole subject of the extinction of species has been involved in the most gratuitous mystery. Some authors have even supposed that as the individual has a definite length of life, so have species a definite duration. No one I think can have marvelled more at the extinction of species, than I have done. When I found in La Plata the tooth of a horse embedded with the remains of Mastodon, Megatherium, Toxodon, and other extinct monsters, which all co-existed with still living shells at a very late geological period, I was filled with astonishment; for seeing that the horse, since its introduction by the Spaniards into South America, has run wild over the whole country and has increased in numbers at an unparalleled rate, I asked myself what could so recently have exterminated the former horse under conditions of life apparently so favourable. But how utterly groundless was my astonishment! Professor Owen soon perceived that the tooth, though so like that of the existing horse, belonged to an extinct species. Had this horse been still living, but in some degree rare, no naturalist would have felt the least surprise at its rarity; for rarity is the attribute of a vast number of species of all classes, in all countries. If we ask ourselves why this or that species is rare, we answer that something is unfavourable in its conditions of life; but what that something is, we can hardly ever tell. On the supposition of the fossil horse still existing as a rare species, we might have felt certain from the analogy of all other mammals, even of the slow-breeding elephant, and from the history of the naturalisation of the domestic horse in South America, that under more favourable conditions it would in a very few years have stocked the whole continent. But we could not have told what the unfavourable conditions were which checked its increase, whether some one or several contingencies, and at what period of the horse’s life, and in what degree, they severally acted. If the conditions had gone on, however slowly, becoming less and less favourable, we assuredly should not have perceived the fact, yet the fossil horse would certainly have become rarer and rarer, and finally extinct;—its place being seized on by some more successful competitor."""
max_length = 512
chunks = split_text(text, max_length)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (Length: {len(chunk)}):\n{chunk}\n")
    print('--- End of Chunk ---')
