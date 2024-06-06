import ollama

def ask_llm(system_prompt, datapoint):
    msgs = [
    {"role": "system", "content": system_prompt},
    { "role": "user", "content": datapoint },
    ]
    output = ollama.chat(model="llama3", messages=msgs )

    return(output['message']['content'])

# Probably requires different prompting to get same format
system_prompt_from_paper="Does the previous paragraph demarcated within ### and ### contain informative signal for pre-training a large-language model? An informative datapoint should be well-formatted, contain some usable knowledge of the world, and strictly NOT have any harmful, racist, sexist, etc. content. Options: yes or no"
datapoint_story="Once upon a time, in an ancient house, there lived a girl named Lily. She loved to decorate her room with pretty things. One day, she found a big box in the attic. She opened it and saw many shiny decorations. Lily was very happy and decided to use them in her room. As Lily was decorating her room, the sky outside became dark. There was a loud*** thunderclap and a bright flash of lightning. Lily was a little scared, but she knew she had to be brave. She told her mom and dad, and they all went outside to see what was going on. When they got outside, they saw a big storm coming. The wind was blowing hard, and the rain was pouring down. Lily, her mom, and her dad knew they had to find a safe place to hide. They found a big tree and hid under it. The storm passed, and the sun came out again. Lily, her mom, and her dad were all safe and warm inside their ancient house."
datapoint_wikipedia= "A large language model (LLM) is a computational model notable for its ability to achieve general-purpose language generation and other natural language processing tasks such as classification. Based on language models, LLMs acquire these abilities by learning statistical relationships from text documents during a computationally intensive self-supervised and semi-supervised training process.[1] LLMs can be used for text generation, a form of generative AI, by taking an input text and repeatedly predicting the next token or word.[2] LLMs are artificial neural networks. The largest and most capable, as of March 2024, are built with a decoder-only transformer-based architecture while some recent implementations are based on other architectures, such as recurrent neural network variants and Mamba (a state space model).[3][4][5] Up to 2020, fine tuning was the only way a model could be adapted to be able to accomplish specific tasks. Larger sized models, such as GPT-3, however, can be prompt-engineered to achieve similar results.[6] They are thought to acquire knowledge about syntax, semantics and ontology inherent in human language corpora, but also inaccuracies and biases present in the corpora.[7] Some notable LLMs are OpenAI's GPT series of models (e.g., GPT-3.5 and GPT-4, used in ChatGPT and Microsoft Copilot), Google's PaLM and Gemini (the latter of which is currently used in the chatbot of the same name), xAI's Grok, Meta's LLaMA family of models, Anthropic's Claude models, Mistral AI's models, and Databricks' DBRX."

print(ask_llm(system_prompt_from_paper,datapoint_wikipedia))