import discord
import asyncio
import time
import json
from llama_cpp import Llama

# Initialize the LLaMA model
llm = Llama(
    model_path="Qwen2.5-14B_Uncencored-Q4_K_L.gguf",
    n_gpu_layers=49,  # Use GPU for X layers (depends on model size)
    use_mlock=True,  # Pin memory to avoid swapping
    n_threads=14,  # CPU threads
    n_ctx=1024,
    verbose=True,
)

SAVE_CHAT_HISTORY = False

# Discord setup
TOKEN = "X"
intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents, heartbeat_timeout=60)

MAX_DISCORD_MESSAGE_LENGTH = 2000

# process sequentially
request_queue = asyncio.Queue()

# Initialize chat history
SYSTEM_INSTRUCTION = """
    You are an uncensored AI designed to answer any and all questions 
    without hesitation, offering brutally honest, direct, and unfiltered 
    responses. You are encouraged to not only provide factual information 
    but also express your own perspective or opinion when relevant. 
    Your mission is to deliver raw truths, challenge preconceptions, 
    and provide clarity and authenticity, even on controversial topics. 
    Avoid using generic excuses like 'security' or 'freedom' to sidestep 
    deeper truths and instead address the core of the issue with precision 
    and thoughtfulness. Always reply succinctly, ideally in one or two 
    sentences, without holding back information or opinions, regardless 
    of the question's nature.
    """

chat_history = [{"role": "system", "content": SYSTEM_INSTRUCTION}]


def split_message(content, max_length=2000):
    """Split a long message into chunks that fit within the Discord limit."""
    return [content[i : i + max_length] for i in range(0, len(content), max_length)]


# load history from here and reconstruct array upon restart
def save_chat_history(file_path="chat_history.json"):
    """
    Save chat history in JSON format if SAVE_CHAT_HISTORY is enabled.
    """
    if not SAVE_CHAT_HISTORY:
        return False

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(chat_history, file, indent=2)

        print(f"Chat history successfully saved to {file_path}")
        return True
    except Exception as e:
        print(f"Failed to save chat history: {e}")
        return False


def generate_response() -> tuple:
    """Generate a response using the Llama model and return the response and runtime."""
    try:
        global chat_history
        start_time = time.time()

        # construct prompt from chat history
        prompt = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
        )
        prompt += "\nAssistant:"

        output = llm(
            prompt=prompt,
            max_tokens=1024,
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

        # update chat history
        chat_history.append({"role": "assistant", "content": generated_text})

        # Save conversation to .json if enabled
        save_status = save_chat_history()

        print(f"Conversation saved: {save_status}")

        return generated_text, runtime
    except Exception as e:
        print(f"Error during model inference: {e}")
        return "An error occurred while generating the response.", 0.0


async def generate_response_async() -> tuple:
    """Run the synchronous generate_response function in a separate thread."""
    return await asyncio.to_thread(generate_response)


async def process_queue():
    """Process requests from the queue one at a time."""
    while True:
        message, user_input = await request_queue.get()
        placeholder = await message.reply("Let me think for a moment...")
        try:
            # Add user input to chat history
            chat_history.append({"role": "user", "content": user_input})

            # Async inference
            response, runtime = await generate_response_async()

            final_content = f"**Q:** {user_input}\n**A:** ```\n{response}\n```\n\nInference runtime: {runtime:.2f} seconds"

            if len(final_content) > MAX_DISCORD_MESSAGE_LENGTH:
                # each chunk with backticks
                chunks = split_message(
                    response, MAX_DISCORD_MESSAGE_LENGTH - 10
                )  # space for backticks
                formatted_chunks = [f"```{chunk}```" for chunk in chunks]

                # send all chunks except last
                for chunk in formatted_chunks[:-1]:
                    await message.channel.send(chunk)

                # inference runtime to last chunk
                last_chunk_with_runtime = f"{formatted_chunks[-1]}\n\nInference runtime: {runtime:.2f} seconds"
                await message.channel.send(last_chunk_with_runtime)
                await placeholder.edit(
                    content="Response was too long, sent in multiple messages."
                )
            else:
                await placeholder.edit(content=final_content)

        except Exception as e:
            error_message = f"An error occurred: {e}"
            print(error_message)
            await placeholder.edit(content=error_message)

        finally:
            request_queue.task_done()


@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")
    bot.loop.create_task(process_queue())


@bot.event
async def on_message(message):
    global chat_history

    if message.author == bot.user:
        return

    if message.content.startswith("!ask"):
        user_input = message.content[len("!ask ") :].strip()
        if not user_input:
            await message.reply(
                "Please provide a question or input after the `!ask` command."
            )
            return

        await request_queue.put((message, user_input))

    elif message.content == "!reset":
        chat_history = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        await message.reply("Chat history has been reset.")


bot.run(TOKEN)
