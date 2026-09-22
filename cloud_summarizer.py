import os
import time
from typing import Optional
from preprocessor import TextPreprocessor

class CloudLLMSummarizer:
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY" if self.provider == "gemini" else "OPENAI_API_KEY")
        self.model_name = model_name or ("gemini-1.5-flash" if self.provider == "gemini" else "gpt-4o-mini")

        if not self.api_key:
            raise ValueError(f"API key for {self.provider} was not provided.")

        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=(
                    "You are an expert literary analyst and book summarizer. "
                    "Extract key ideas, narrative arcs, and major themes accurately."
                )
            )
        elif self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)

    def _build_prompt(self, text: str, length_setting: str, style: str, title: str, author: str) -> str:
        length_guidelines = {
            "Concise": "approximately 150-250 words capturing key points",
            "Medium": "approximately 400-600 words with balanced coverage",
            "Detailed": "approximately 800-1200 words with in-depth analysis"
        }
        target_length = length_guidelines.get(length_setting, "400-600 words")
        format_instruction = (
            "Format the summary as organized bullet points with bold subheadings."
            if style == "Bullet Points"
            else "Format the summary in coherent, well-structured paragraphs with clear transitions."
        )
        author_str = f" by {author}" if author else ""
        metadata = f"Book: {title}{author_str}"

        return (
            f"Please analyze and summarize the following text from {metadata}.\n\n"
            f"Requirements:\n"
            f"1. Target Length: {target_length}.\n"
            f"2. Output Format: {format_instruction}\n"
            f"3. Content Focus:\n"
            f"   - Central premise or thesis.\n"
            f"   - Core supporting events, themes, or arguments.\n"
            f"   - Key takeaways and conclusions.\n\n"
            f"Text to summarize:\n"
            f"---------------------\n"
            f"{text}\n"
            f"---------------------\n"
        )

    def summarize_single_pass(self, text: str, length_setting: str, style: str, title: str = "Untitled", author: str = "") -> str:
        prompt = self._build_prompt(text, length_setting, style, title, author)
        if self.provider == "gemini":
            import google.generativeai as genai
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2048
                )
            )
            return response.text.strip()
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert book summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()

    def summarize_map_reduce(self, text: str, length_setting: str, style: str, title: str = "Untitled", author: str = "") -> str:
        chunks = TextPreprocessor.create_overlapping_chunks(text, chunk_size=2500, overlap=250)
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"Summarize key ideas and main events from this section of {title} in 100-150 words:\n\n{chunk}"
            if self.provider == "gemini":
                res = self.client.generate_content(chunk_prompt)
                chunk_summaries.append(res.text.strip())
            time.sleep(0.5)

        combined_input = "\n\n".join([f"Section {i+1} Summary:\n{s}" for i, s in enumerate(chunk_summaries)])
        return self.summarize_single_pass(combined_input, length_setting, style, title, author)
