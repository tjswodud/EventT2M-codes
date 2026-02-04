import os
import google.generativeai as genai

from tqdm import tqdm

GOOGLE_API_KEY = "" # your api key here
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

def decompose(lines):
    generated_output = []
    for j, line in enumerate(lines):
        prompt = f"""
        Please segment a single input sentence into multiple sentences that each represent a distinct event, following the rules below:

        - A bundle of actions performed simultaneously at a specific point in time is defined as a single “event.”
        - Each segmented sentence must start with the subject used in the original sentence (e.g., "a person", "a man", etc.).
        - Do not remove or simplify any adverbs, adjectives, or modifiers that appear in the original sentence — preserve them as much as possible.
        - Parts separated by the # symbol must be included in each segmented sentence.
        - Do not add any new sentences — only break down the given text as instructed.
        - Do not include your thinking process or output reasoning. Only output the segmented sentences following the format.
        - Even if there are grammatical errors in the sentence, proceed with the processing.
        - If the input sentence contains multiple actions, the output must contain same number of actions as the input sentence.

        [Good Example 1 - Input]
        a man lifts something on his left and places it down on his right.#a/DET man/NOUN lift/VERB something/PRON on/ADP his/DET left/NOUN and/CCONJ place/VERB it/PRON down/ADP on/ADP his/DET right/NOUN#0.0#0.0

        [Good Example 1 - Output]
        a man lifts something on his left.#a/DET man/NOUN lift/VERB something/PRON on/ADP his/DET left/NOUN#0.0#0.0
        a man places it down on his right.#a/DET man/NOUN place/VERB it/PRON down/ADP on/ADP his/DET right/NOUN#0.0#0.0

        [Good Example 2 - Input]
        a man kicks something with his left leg.#a/DET man/NOUN kick/VERB something/PRON with/ADP his/DET left/ADJ leg/NOUN#0.0#0.0

        [Good Example 2 - Output]
        a man kicks something with his left leg.#a/DET man/NOUN kick/VERB something/PRON with/ADP his/DET left/ADJ leg/NOUN#0.0#0.0

        [Good Example 3 - Input]
        a person waves their hand while stepping sideways, then jumps up and spins, and finally lands and bows. #a/DET person/NOUN wave/VERB their/DET hand/NOUN while/SCONJ step/VERB sideways/ADV then/ADV jump/VERB up/ADV and/CCONJ spin/VERB and/CCONJ finally/ADV land/VERB and/CCONJ bow/VERB#0.0#0.0

        [Good Example 3 - Output]
        a person waves their hand while stepping sideways.#a/DET person/NOUN wave/VERB their/DET hand/NOUN while/SCONJ step/VERB sideways/ADV#0.0#0.0
        a person jumps up and spins.#a/DET person/NOUN jump/VERB up/ADV and/CCONJ spin/VERB#0.0#0.0
        a person lands and bows.#a/DET person/NOUN land/VERB and/ CCONJ bow/VERB#0.0#0.0

        [Bad Example 1 - Input]
        someone is sprinting side to side#someone/PRON is/AUX sprint/VERB side/NOUN to/PART side/VERB#0.0#0.0

        [Bad Example 1 - Output]
        The input sentence cannot be segmented as requested because "side to side" is a single adverbial phrase modifying the verb "sprinting," and "side" is used as a noun and a prepositional phrase.  The sentence only contains one action (sprinting).

        [Bad Example 2 - Input]
        a person claps their hands while sitting on the ground. #a/DET person/NOUN clap/VERB their/DET hands/NOUN while/SCONJ sit/VERB on/ADP the/DET ground/NOUN#0.0#0.0

        [Bad Example 2 - Output]
        a person claps their hands. #a/DET person/NOUN clap/VERB their/DET hands/NOUN#0.0#0.0
        a person sits on the ground. #a/DET person/NOUN sit/VERB on/ADP the/DET ground/NOUN#0.0#0.0

        [Bad Example 3 - Input]
        a person bends their knees and raises both arms at the same time.#a/DET person/NOUN bend/VERB their/DET knees/NOUN and/CCONJ raise/VERB both/DET arms/NOUN at/ADP the/DET same/ADJ time/NOUN#0.0#0.0

        [Bad Example 3 - Output]
        a person bends their knees.#a/DET person/NOUN bend/VERB their/DET knees/NOUN#0.0#0.0
        a person raises both arms.#a/DET person/NOUN raise/VERB both/DET arms/NOUN#0.0#0.0

        [Input]
        {line}
        """
        
        response = model.generate_content(prompt)
        if response.candidates and response.candidates[0].content.parts:
            generated_output.append(response.text)
        else:
            print("There is no valid content in the response.")

    return generated_output

def process(file_path):
    print(f"Processing data...")
    with open(os.path.join(file_path, "test"+".txt"), "r") as f:
        id_list = [line.strip() for line in f.readlines()]

    for i, x in enumerate(tqdm(id_list)):
        text_path = os.path.join(file_path, "texts", x+".txt")
        with open(text_path) as f:
            lines = f.readlines()

        output = decompose(lines)

        with open(os.path.join(file_path, "texts", x + ".txt"), "w") as f:
            for line in output:
                f.write(line)
                f.write("\n\n")

def main():
    file_path = "./data/HumanML3D/"

    process(file_path)

if __name__ == "__main__":
    print(f"start dataset preparation...")
    main()
    print(f"dataset preparation done!")