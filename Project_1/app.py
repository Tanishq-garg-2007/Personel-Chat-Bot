from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr

load_dotenv(override=True)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
google_api_key = os.getenv("GOOGLE_API_KEY")

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string"},
            "name": {"type": "string"},
            "notes": {"type": "string"}
        },
        "required": ["email"],
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
        },
        "required": ["question"],
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]

class Me:
    def __init__(self):
        self.gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)  # ✅ fixed
        self.name = "Ed Donner"
        self.linkedin = ""
        try:
            reader = PdfReader("me/New Resume 12e9d8.pdf")
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    self.linkedin += text
        except FileNotFoundError:
            self.linkedin = "LinkedIn profile not found."
        try:
            with open("me/summary.txt", "r", encoding="utf-8") as f:
                self.summary = f.read()
        except FileNotFoundError:
            self.summary = "Summary not found."

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results

    def system_prompt(self):
        return f"You are {self.name}. Answer professionally.\n\n## Summary:\n{self.summary}\n\n## LinkedIn:\n{self.linkedin}"

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.gemini.chat.completions.create(
                model="gemini-2.5-flash-preview-05-20",
                messages=messages,
                tools=tools
            )
            choice = response.choices[0]
            if choice.finish_reason == "tool_calls":
                tool_calls = choice.message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(choice.message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message["content"]  # ✅ fixed

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(fn=me.chat).launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))


