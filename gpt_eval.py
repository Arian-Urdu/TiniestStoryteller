import ollama

# check https://github.com/ollama/ollama on how to install llama3

def gpt_eval(system_prompt, story):
    msgs = [
    {"role": "system", "content": system_prompt},
    { "role": "user", "content": story },
    ]
    output = ollama.chat(model="llama3", messages=msgs )

    return(output['message']['content'])

# Probably requires different prompting to get same format
system_prompt_from_paper="the following exercise, the student is given a beginning of a story. The student needs to complete it into a full story. The exercise tests the student's language abilities and creativity. The symbol *** marks the separator between the prescribed beginning and the student's completion. Please provide your general assessment about the part written by the student (the one after the *** symbol). Is it gramatically correct? Is it consistent with the beginning of the story? Pay special attention to whether the student manages to complete the sentence which is split in the middle by the separator ***. Also, grade the student’s completion in terms of grammar from 0 to 10, creativity from 0 to 10, consistency with the story’s beginning and whether the plot makes sense from 0 to 10. And finally,please provide your best guess of what the age of the student might be, as reflected from the completion. Choose from possible age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16.  "
story_example_paper="Once upon a time, in an ancient house, there lived a girl named Lily. She loved to decorate her room with pretty things. One day, she found a big box in the attic. She opened it and saw many shiny decorations. Lily was very happy and decided to use them in her room. As Lily was decorating her room, the sky outside became dark. There was a loud*** thunderclap and a bright flash of lightning. Lily was a little scared, but she knew she had to be brave. She told her mom and dad, and they all went outside to see what was going on. When they got outside, they saw a big storm coming. The wind was blowing hard, and the rain was pouring down. Lily, her mom, and her dad knew they had to find a safe place to hide. They found a big tree and hid under it. The storm passed, and the sun came out again. Lily, her mom, and her dad were all safe and warm inside their ancient house."
story_example_1= "Once upon a time there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, shiny rock in the sky. She wanted to touch it, but it was too high. Lily's mommy told her that it was important to be careful and not touch things. Lily didn't want to touch it, so she asked her mommy if she could touch it. Her mommy said yes and they went to the park. When they got there, Lily saw a big, scary dog. The dog was barking and barked loudly. Lily was scared and didn't know what to do. She tried to run away, but she was too fast. She tried to run away, but it was too late. The dog was too fast and it was too fast. Lily was sad and cried. She wished she had listened to her mommy. She wished she had listened to her mommy and never let her to get back."

print(gpt_eval(system_prompt_from_paper,story_example_1))

