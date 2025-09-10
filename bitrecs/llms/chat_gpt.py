from openai import OpenAI

class ChatGPT:
    def __init__(self, 
                 key,
                 model="gpt-3.5-turbo", 
                 system_prompt="You are a helpful assistant.", 
                 temp=0.0):
        
        self.CHATGPT_API_KEY = key
        if not self.CHATGPT_API_KEY:
            raise ValueError("CHATGPT_API_KEY is not set")
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        # Create client once during initialization for speed
        self.client = OpenAI(api_key=self.CHATGPT_API_KEY)

    def call_chat_gpt(self, prompt) -> str:
        if not prompt or len(prompt) < 10:
            raise ValueError(f"Prompt too short: {len(prompt) if prompt else 0} characters (minimum 10)")

        if not self.CHATGPT_API_KEY:
            raise ValueError("CHATGPT_API_KEY is not set")

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
            {
                "role": "user",
                "content": prompt,
            }],
            temperature=self.temp,
            max_tokens=512,  # Increased for better quality responses
            timeout=3.0,  # Keep timeout for speed
            stream=False,  # Disable streaming for faster completion
            top_p=0.3,  # Increased for better creativity and diversity
            frequency_penalty=0.0,  # No frequency penalty
            presence_penalty=0.0  # No presence penalty
        )

        thing = completion.choices[0].message.content                
        return thing