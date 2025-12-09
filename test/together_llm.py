from together import Together

client = Together(api_key="")

response = client.chat.completions.create(
    model="ServiceNow-AI/Apriel-1.6-15b-Thinker",
    messages=[
      {
        "role": "user",
        "content": "What are some fun things to do in New York?"
      }
    ]
)
print(response.choices[0].message.content)