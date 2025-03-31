# tg_parsing.py
# This script uses Telethon to fetch messages from a public Telegram channel.
# You need to install Telethon first: pip install telethon pandas
#
# IMPORTANT:
# 1. You MUST get your own api_id and api_hash from my.telegram.org.
#    Do NOT use placeholder values.
# 2. Running this script will require you to log in with your phone number
#    and a code sent by Telegram the first time you run it. You might also
#    need your two-step verification password if enabled.
# 3. Be mindful of Telegram's Terms of Service regarding automation and scraping.
#    Excessive requests might lead to temporary or permanent limitations (FloodWaitError).
# 4. This script saves data to a CSV file named 'progressagro_messages.csv'.
# 5. NEVER share your api_id, api_hash, or the generated session file ('tg_session.session').

import asyncio
import os
import random

import pandas as pd  # Using pandas for easier CSV handling later
from telethon.errors import FloodWaitError, SessionPasswordNeededError
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import MessageMediaDocument, MessageMediaPhoto, MessageMediaWebPage

# --- Configuration ---
# Replace with your own API credentials obtained from my.telegram.org
# Consider using environment variables or a more secure config method in production.
API_ID = 1234567  # !!! REPLACE WITH YOUR API ID !!!
API_HASH = "YOUR_API_HASH"  # !!! REPLACE WITH YOUR API HASH !!!
PHONE = "+12345678900"  # !!! REPLACE WITH YOUR PHONE NUMBER (international format) !!!

# Channel to parse
CHANNEL_LINK = "https://t.me/progressagro" # Or just 'progressagro' if it's a public username

# Output file
OUTPUT_CSV_FILE = "progressagro_messages.csv"

# Session file name
SESSION_FILE = "tg_session" # Will create tg_session.session

# --- Helper Function to handle None values for CSV ---
def none_to_empty_string(value):
    """Converts None to an empty string, otherwise returns the value."""
    return "" if value is None else value

# --- Main Asynchronous Function ---
async def main():
    # Handling potential asyncio issues on Windows at the start of main
    # Moved from if __name__ == "__main__": block
    if os.name == "nt":
         try:
              # Attempt to set the policy only if needed and available
              # Check if the necessary policy exists before trying to set it
              if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
                   current_policy = asyncio.get_event_loop_policy()
                   # Set it only if it's not already the correct type
                   if not isinstance(current_policy, asyncio.WindowsSelectorEventLoopPolicy):
                        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                        print("Applied Windows asyncio policy.")
              else:
                   print("WindowsSelectorEventLoopPolicy not available on this asyncio version.")
         except Exception as e:
              # Catch potential errors during policy setting
              print(f"Warning: Could not set Windows asyncio policy: {e}")

    print("Initializing Telegram Client...")
    # Create the client instance
    client = TelegramClient(SESSION_FILE, API_ID, API_HASH, system_version="4.16.30-vxCUSTOM")

    try:
        print(f"Connecting to Telegram (using phone: {PHONE})...")
        await client.connect()

        # Authorize the user if necessary
        if not await client.is_user_authorized():
            print("First run or session expired: Sending code request...")
            await client.send_code_request(PHONE)
            try:
                code = input("Enter the code you received from Telegram: ")
                await client.sign_in(PHONE, code)
            except SessionPasswordNeededError:
                password = input("Two-step verification enabled. Please enter your password: ")
                await client.sign_in(password=password)
            print("Signed in successfully!")
        else:
            print("Already authorized.")

        print(f"Accessing channel: {CHANNEL_LINK}")
        try:
            # Get the channel entity
            entity = await client.get_entity(CHANNEL_LINK)
            print(f"Successfully accessed channel: {getattr(entity, 'title', 'Unknown Title')}")
        except ValueError as e:
            print(f"Error: Could not find the channel '{CHANNEL_LINK}'. Please check the username or link. Details: {e}")
            return
        except Exception as e:
            print(f"An unexpected error occurred while getting the channel entity: {e}")
            return

        print("Starting to fetch messages...")

        all_messages_data = [] # List to store message data dictionaries
        offset_id = 0
        limit = 100  # Number of messages to fetch per request (max 100)
        total_messages_fetched = 0
        # Limit total messages to avoid excessive requests (adjust as needed, e.g., 10000)
        # Set to 0 to fetch all available messages (USE WITH CAUTION!)
        max_messages_to_fetch = 0

        while True:
            if max_messages_to_fetch != 0 and total_messages_fetched >= max_messages_to_fetch:
                 print(f"Reached fetch limit of {max_messages_to_fetch} messages.")
                 break

            print(f"Fetching batch starting from offset_id: {offset_id} (Total fetched so far: {total_messages_fetched})")
            try:
                history = await client(GetHistoryRequest(
                    peer=entity,
                    offset_id=offset_id,
                    offset_date=None,
                    add_offset=0,
                    limit=limit,
                    max_id=0,
                    min_id=0,
                    hash=0
                ))
            except FloodWaitError as e:
                print(f"Flood wait error: Need to wait for {e.seconds} seconds.")
                print(f"Sleeping for {e.seconds + 5} seconds...") # Wait a bit longer
                await asyncio.sleep(e.seconds + 5)
                continue # Retry the same batch
            except Exception as e:
                print(f"Error fetching message history: {e}")
                print("Waiting for 60 seconds before retrying...")
                await asyncio.sleep(60)
                continue # Retry the same batch

            if not history.messages:
                print("No more messages found.")
                break  # Exit loop if no messages are left

            messages_in_batch = 0
            for message in history.messages:
                messages_in_batch += 1
                # Basic info
                msg_id = message.id
                date_utc = message.date # Keep as datetime object for now
                text = message.message if message.message else ""
                views = message.views if hasattr(message, "views") else None

                # Sender info (might be None for channel posts if posted anonymously or by channel itself)
                sender_id = None
                sender_username = None
                # Check message.sender first (for users in linked chats/comments perhaps)
                if message.sender_id:
                     sender_id = message.sender_id
                     # Try getting sender entity if needed (can be slow, use cautiously)
                     # sender_entity = await client.get_entity(sender_id)
                     # sender_username = getattr(sender_entity, 'username', None)


                # Media info extraction
                has_media = message.media is not None
                media_type = None
                file_name = None
                file_size_bytes = None
                is_voice = False
                is_video = False
                duration = None

                if isinstance(message.media, MessageMediaPhoto):
                    media_type = "photo"
                    # Getting photo size requires more effort, maybe skip for simplicity or add later
                elif isinstance(message.media, MessageMediaDocument):
                    media_type = "document"
                    doc = message.media.document
                    file_size_bytes = getattr(doc, "size", None)
                    mime_type = getattr(doc, "mime_type", "").lower()

                    # Try to get filename and refine media type
                    for attr in getattr(doc, "attributes", []):
                        if hasattr(attr, "file_name"):
                            file_name = attr.file_name
                        if hasattr(attr, "duration"):
                            duration = attr.duration
                        if hasattr(attr, "voice") and attr.voice:
                            is_voice = True
                        if hasattr(attr, "round_message") and attr.round_message:
                             media_type = "video_message" # Treat round messages separately
                             is_video = True # It is a video type
                        elif hasattr(attr, "supports_streaming") and attr.supports_streaming:
                            is_video = True # Streaming usually means video

                    # Refine media type based on mime type or attributes
                    if is_voice:
                        media_type = "voice"
                    elif is_video or "video" in mime_type:
                         media_type = "video"
                    elif "audio" in mime_type:
                         media_type = "audio"
                    elif file_name: # If it has a filename and not identified otherwise
                         media_type = f"document ({os.path.splitext(file_name)[1]})" # Use extension
                    else:
                         media_type = f"document ({mime_type})"

                elif isinstance(message.media, MessageMediaWebPage):
                    media_type = "webpage"
                elif message.media is not None: # Catch other media types if any
                     media_type = str(type(message.media)).split(".")[-1].replace("MessageMedia", "").lower()


                # Store data
                all_messages_data.append({
                    "message_id": msg_id,
                    "date_utc": date_utc,
                    "sender_id": sender_id,
                    # 'sender_username': sender_username, # Add back if needed
                    "text": text,
                    "has_media": has_media,
                    "media_type": media_type,
                    "file_name": file_name,
                    "file_size_bytes": file_size_bytes,
                    "duration_seconds": duration,
                    "views": views
                })

            total_messages_fetched += messages_in_batch
            if not history.messages: # Check again just in case
                 break
            offset_id = history.messages[-1].id # ID of the oldest message in the batch
            print(f"Fetched {messages_in_batch} messages. Oldest message ID in batch: {offset_id}")

            # Optional: Add a small delay to be polite to Telegram servers
            wait_time = random.uniform(1.5, 3.5) # Случайная задержка от 1.5 до 3.5 секунд
            print(f"Waiting for {wait_time:.2f} seconds before next batch...")
            await asyncio.sleep(wait_time)

        print(f"\nFinished fetching messages. Total fetched: {total_messages_fetched}")

        # --- Save data to CSV using pandas ---
        if all_messages_data:
            print("Converting data to DataFrame...")
            df = pd.DataFrame(all_messages_data)
            # Convert datetime to string for CSV
            df["date_utc"] = df["date_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
            # Fill NaN values if any (important for CSV writing)
            df.fillna("", inplace=True)

            print(f"Saving data to {OUTPUT_CSV_FILE}...")
            # Save to CSV, overwrite if exists
            df.to_csv(OUTPUT_CSV_FILE, index=False, encoding="utf-8")
            print("Data saved successfully.")
        else:
            print("No messages were fetched or processed.")


    except Exception as e:
        print(f"\nAn critical error occurred: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        if client.is_connected():
            print("Disconnecting client...")
            await client.disconnect()
        print("Script finished.")

# --- Run the script ---
if __name__ == "__main__":
    # The OS-specific check is now inside main()
    asyncio.run(main())