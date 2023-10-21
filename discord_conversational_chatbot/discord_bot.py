# LIBRARIES
import os
import discord
from discord.ext import commands

from langchain import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# APIKEY AND TOKENS
from token import token
os.environ['HUGGINGFACEHUB_API_TOKEN'] = token

from discord_token import discord_token
DISCORD_TOKEN = discord_token

# LLM INITIALIZATION
model_id = "lmsys/fastchat-t5-3b-v1.0"
conv_model = HuggingFaceHub(huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN'],
                            repo_id = model_id,
                            model_kwargs = {"temperature" : 0.8,
                                            "max_new_tokens" : 150})
conversation = ConversationChain(llm=conv_model)

memory = ConversationBufferMemory()
conversation_buf = ConversationChain(llm=conv_model, memory=memory)
queries_list = []

# DISCORD BOT INITIALIZATION
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
bot = commands.Bot(command_prefix='!', intents=intents)

# BOT EVENT & COMMANDS
@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')

@bot.command(name='gpt')
async def say_hi(ctx):
    input = ctx.message.content
    input = input.split(' ', 1)[1]
    print(input)

    response = conversation_buf.predict(input=input)
    response = response.split(' ', 1)[1]
    print(response)

    await ctx.send(response)

bot.run(DISCORD_TOKEN)