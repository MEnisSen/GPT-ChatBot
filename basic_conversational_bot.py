import os
from token import token
from langchain import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# APIKEY for Hugging Face
os.environ['HUGGINGFACEHUB_API_TOKEN'] = token

# Model selection
model_id = "lmsys/fastchat-t5-3b-v1.0"
#model_id = "gpt2-medium"

# Model initialization
conv_model = HuggingFaceHub(huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN'],
                            repo_id = model_id,
                            model_kwargs = {"temperature" : 0.8,
                                            "max_new_tokens" : 32})

# Adding the ability to hold a conversation with memory held of previous prompts
conversation = ConversationChain(llm=conv_model)
memory = ConversationBufferMemory()
conversation_buf = ConversationChain(llm=conv_model, memory=memory)

# Can be arranged based on the conversation length or style
for i in range(10):
    print(conversation_buf.predict(input=input("Input your prompt :\n")))