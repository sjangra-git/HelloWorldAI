import json
import os
from datetime import datetime
from openai import OpenAI

# Initialize the client using environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running this script.")

client = OpenAI(api_key=api_key)

# 1. Define the actual Python function (The Tool)
def get_current_time():
    """Returns the current date and time as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 2. Tell the LLM that this tool exists
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Use this tool to get the live, exact current date and time.",
            "parameters": {
                "type": "object",
                "properties": {}, # No inputs needed for this function
            },
        },
    }
]

def ask_agent(user_prompt):
    print(f"User Question: '{user_prompt}'")
    
    # Pack the message history
    messages = [{"role": "user", "content": user_prompt}]
    
    # First turn: Send the question and tool definitions to the LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )
    
    assistant_message = response.choices[0].message
    
    # Check if the LLM decided it needs to use our tool
    if assistant_message.tool_calls:
        tool_call = assistant_message.tool_calls[0]
        function_name = tool_call.function.name
        
        print(f"🤖 Agent Status: Decided to call tool -> '{function_name}()'")
        
        # Execute the native Python function based on the LLM's decision
        if function_name == "get_current_time":
            print(f"📋 Search Results: function_name = '{function_name}', tool_call = {tool_call}, parameters = {tool_call.function.arguments}")
            tool_result = get_current_time()
            print(f"🔧 Tool Output: Generated result -> '{tool_result}'")
            
            # Append the LLM's intent and our tool's actual output to the conversation
            messages.append(assistant_message)
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": tool_result
            })
            
            # Second turn: Send everything back so the LLM can formulate a final response
            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return final_response.choices[0].message.content
    else:
        # If the LLM didn't need the tool, just return its text answer
        print("🤖 Agent Status: Answered directly without tools.")
        return assistant_message.content

# --- TEST THE AGENT ---
print("--- Test 1: Needs a Tool ---")
print(f"Final Answer: {ask_agent('What exact time is it right now?')}\n")

print("--- Test 2: Does NOT Need a Tool ---")
print(f"Final Answer: {ask_agent('What is the capital of France?')}")
