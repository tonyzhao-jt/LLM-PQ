# chatgpt generated input tokens
prompts_128 = prompts = [
    "What is he working on?",
    "What does he have?",
    "How can I make everyone happy?",
    "What are you planning to do?",
    "What is your favorite hobby?",
    "What do you think about the latest news?",
    "How can we improve our community?",
    "What is the meaning of life?",
    "What are your goals for the future?",
    "What is the most interesting place you have ever visited?",
    "What is the best way to learn a new skill?",
    "What do you think about the future of technology?",
    "How can we reduce our impact on the environment?",
    "What is the most important lesson you have learned in life?",
    "What is the best book you have ever read?",
    "What is your favorite type of music?",
    "What do you like to do in your free time?",
    "How can we promote equality and diversity?",
    "What is the key to a successful career?",
    "What do you think is the biggest challenge facing our society today?"
]

prompts_512 = [
    "In a world where technology has advanced beyond our wildest dreams, humans are living in a utopia. They have access to all the resources they need and are free to pursue their passions without any hindrance. However, there is a group of rebels who are determined to overthrow the government and establish a new order. They believe that the current system is corrupt and needs to be replaced. Their leader is a charismatic young woman named",
    "The year is 2077 and humanity has colonized the entire solar system. They have built massive space stations and terraformed the planets to make them habitable. However, there are still dangers lurking in the void of space. The alien race known as the Zorgons are determined to wipe out humanity and claim the solar system for themselves. It's up to a team of brave astronauts to",
    "She walked into the room and saw a sight that took her breath away. The walls were covered in intricate murals depicting scenes from ancient mythology. The floor was made of polished marble, and in the center of the room stood a magnificent statue of a",
    "The sound of rain hitting the window was soothing. It reminded her of lazy days spent curled up with a good book. She sipped her tea and watched as the raindrops traced their way down the glass. She wondered where the rain was coming from, and if there were other people in different parts of the world who were also",
    "As the sun set over the horizon, the sky was painted with shades of red and orange. It was a beautiful sight, and it made her feel grateful to be alive. She took a deep breath of the fresh air and felt a sense of peace wash over her. She knew that there were still challenges to face, but for now, she was content to",
    "The spaceship had been adrift in space for months. The crew was running low on supplies and morale was low. They had encountered many challenges along the way, from asteroid storms to hostile alien races. But they were determined to complete their mission and",
    "He had been walking for days and was starting to feel the effects of exhaustion. His feet were sore and his muscles ached. He longed for a warm meal and a comfortable bed, but he knew that he couldn't stop yet. He was on a mission to",
    "The secret to a happy life is simple: follow your heart. Do what you love, and you'll never work a day in your life. It's important to surround yourself with people who support and encourage you, and to always",
    "The storm was raging outside and the wind was howling. She felt a sense of unease as she listened to the sounds of the tempest. She wondered if she was safe inside her home, or if the storm would",
    "The door creaked open and she saw a figure standing in the shadows. She felt a chill run down her spine as she realized that she was not alone. She tried to remain calm, but her heart was racing. She wondered who the stranger was, and what they wanted from",
    "The world was in chaos and people were afraid. The economy had collapsed and resources were scarce. There were riots in the streets and violence was commonplace. But in the midst of all the turmoil, there were still pockets of hope. People were coming together to",
    "The city was bustling with activity and the streets were crowded. There were vendors selling food and drinks, musicians playing music, and children laughing and playing. It was a scene of joy and celebration. She felt a sense of belonging",
]

prompts_others = [
    "In a world where technology has advanced beyond our wildest dreams, humans are living in a utopia. They have access to all the resources they need and are free to pursue their passions without any hindrance. However, there is a group of rebels who are determined to overthrow the government and establish a new order. They believe that the current system is corrupt and needs to be replaced. Their leader is a charismatic young woman named",
    "The year is 2077 and humanity has colonized the entire solar system. They have built massive space stations and terraformed the planets to make them habitable. However, there are still dangers lurking in the void of space. The alien race known as the Zorgons are determined to wipe out humanity and claim the solar system for themselves. It's up to a team of brave astronauts to",
    "She walked into the room and saw a sight that took her breath away. The walls were covered in intricate murals depicting scenes from ancient mythology. The floor was made of polished marble, and in the center of the room stood a magnificent statue of a",
    "The sound of rain hitting the window was soothing. It reminded her of lazy days spent curled up with a good book. She sipped her tea and watched as the raindrops traced their way down the glass. She wondered where the rain was coming from, and if there were other people in different parts of the world who were also",
    "As the sun set over the horizon, the sky was painted with shades of red and orange. It was a beautiful sight, and it made her feel grateful to be alive. She took a deep breath of the fresh air and felt a sense of peace wash over her. She knew that there were still challenges to face, but for now, she was content to",
    "The spaceship had been adrift in space for months. The crew was running low on supplies and morale was low. They had encountered many challenges along the way, from asteroid storms to hostile alien races. But they were determined to complete their mission and",
    "He had been walking for days and was starting to feel the effects of exhaustion. His feet were sore and his muscles ached. He longed for a warm meal and a comfortable bed, but he knew that he couldn't stop yet. He was on a mission to",
    "The secret to a happy life is simple: follow your heart. Do what you love, and you'll never work a day in your life. It's important to surround yourself with people who support and encourage you, and to always",
    "The storm was raging outside and the wind was howling. She felt a sense of unease as she listened to the sounds of the tempest. She wondered if she was safe inside her home, or if the storm would",
    "The door creaked open and she saw a figure standing in the shadows. She felt a chill run down her spine as she realized that she was not alone. She tried to remain calm, but her heart was racing. She wondered who the stranger was, and what they wanted from",
    "The world was in chaos and people were afraid. The economy had collapsed and resources were scarce. There were riots in the streets and violence was commonplace. But in the midst of all the turmoil, there were still pockets of hope. People were coming together to",
    "The city was bustling with activity and the streets were crowded. There were vendors selling food and drinks, musicians playing music, and children laughing and playing. It was a scene of joy and celebration. She felt a sense of belonging",
]


# randomly fetch one token
import random
def fetch_prompts(batch_size=1, prompt_length=128):
    if prompt_length == 128:
        prompts = random.choices(prompts_128, k=batch_size)
    elif prompt_length == 512:
        prompts = random.choices(prompts_512, k=batch_size)
    else:
        prompts = random.choices(prompts_others, k=batch_size)
        # raise ValueError("prompt length should be 128 or 512")
    return prompts