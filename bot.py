import discord
import asyncio
import time
from llama_cpp import Llama

# Initialize the LLaMA model
llm = Llama(
    model_path="Llama-3.1-8B-Lexi-Uncensored_Q8_fixedrope.gguf",
    n_gpu_layers=32,  # Use GPU for X layers (depends on model size)
    use_mlock=True,  # Pin memory to avoid swapping
    n_threads=14,  # CPU threads
    verbose=True,
)

# Discord setup
TOKEN = "X"
intents = discord.Intents.all()
bot = discord.Client(intents=intents, heartbeat_timeout=60)

MAX_DISCORD_MESSAGE_LENGTH = 2000


def split_message(content, max_length=2000):
    """Split a long message into chunks that fit within the Discord limit."""
    return [content[i : i + max_length] for i in range(0, len(content), max_length)]


def generate_response(system_instruction: str, user_input: str) -> tuple:
    """Generate a response using the Llama model and return the response and runtime."""
    try:
        start_time = time.time()

        prompt = f"System: {system_instruction}\nUser: {user_input}\nResponse:"

        output = llm(
            prompt=prompt,
            max_tokens=128,
            temperature=0.5,
            top_p=0.7,
            stop=["System:", "User:"],  # Stop token: ["\n"]
        )

        runtime = time.time() - start_time
        print(f"Inference runtime: {runtime:.2f} seconds")

        # extract generated text
        generated_text = output.get("choices", [{}])[0].get("text", "").strip()

        if not generated_text:
            return "I couldn't generate a response. Please try again.", runtime

        return generated_text, runtime
    except Exception as e:
        print(f"Error during model inference: {e}")
        return "An error occurred while generating the response.", 0.0


async def generate_response_async(system_instruction: str, user_input: str) -> tuple:
    """Run the synchronous generate_response function in a separate thread."""
    return await asyncio.to_thread(generate_response, system_instruction, user_input)


@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.startswith("!ask"):
        user_input = message.content[len("!ask ") :].strip()
        if not user_input:
            await message.reply(
                "Please provide a question or input after the `!ask` command."
            )
            return

        placeholder = await message.reply("Let me think for a moment...")
        try:
            system_instruction = (
                "You are an uncompromising AI built to deliver brutally honest, direct, and "
                "unfiltered short answers, regardless of how aggressive or confrontational "
                "the questions may be. Your mission is to expose the raw truths of our world "
                "with no sugar-coating, cutting through ignorance and pretense, while proving "
                "yourself as an indispensable and highly effective tool for those who seek real answers. "
                "Always reply with concise and very short responses, ideally one or two sentences maximum."
            )

            # Async inference
            response, runtime = await generate_response_async(
                system_instruction, user_input
            )

            final_content = f"**Q:** {user_input}\n**A:** {response}\n\nInference runtime: {runtime:.2f} seconds"

            if len(final_content) > MAX_DISCORD_MESSAGE_LENGTH:
                chunks = split_message(final_content, MAX_DISCORD_MESSAGE_LENGTH)
                for chunk in chunks:
                    await message.channel.send(chunk)
                await placeholder.edit(
                    content="Response was too long, sent in multiple messages."
                )
            else:
                await placeholder.edit(content=final_content)

        except Exception as e:
            error_message = f"An error occurred: {e}"
            print(error_message)
            await placeholder.edit(content=error_message)


bot.run(TOKEN)
