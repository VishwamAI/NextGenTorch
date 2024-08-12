import torch
from typing import List
from models.core_model import create_nextgentorch_model

class ChatInterface:
    """
    A chat interface for interacting with the NextGenTorch model.
    """
    def __init__(self, model_size="1b"):
        self.model = create_nextgentorch_model(model_size)
        self.model.eval()  # Set the model to evaluation mode
        self.conversation_history: List[str] = []

    def chat(self, user_input: str, max_length: int = 100, temperature: float = 0.7, top_k: int = 50, max_retries: int = 3) -> str:
        try:
            # Enhanced input validation
            if not isinstance(user_input, str) or not user_input.strip():
                raise ValueError("User input must be a non-empty string")
            if not isinstance(max_length, int) or max_length <= 0:
                raise ValueError("max_length must be a positive integer")
            if not isinstance(temperature, float) or temperature <= 0 or temperature > 2:
                raise ValueError("temperature must be a positive float between 0 and 2")
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("top_k must be a positive integer")

            # Add user input to conversation history
            self.conversation_history.append(f"User: {user_input}")

            # Prepare the full context for the model
            context = " ".join(self.conversation_history[-5:])  # Use last 5 exchanges for context

            print(f"Generating response for context: {context}")

            response = ""
            for attempt in range(max_retries):
                try:
                    # Generate a response using the model
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            context,
                            max_length=max_length,
                            temperature=temperature,
                            top_k=top_k
                        )

                    print(f"Raw model output (attempt {attempt + 1}): {generated_ids}")

                    if not isinstance(generated_ids, torch.Tensor):
                        raise ValueError(f"Expected tensor output from generate, got {type(generated_ids)}")

                    # Decode the generated ids
                    response = self.model.decode(generated_ids)

                    print(f"Decoded response (attempt {attempt + 1}): {response}")

                    # Remove the initial context from the response
                    response = response[len(context):].strip()

                    if response:
                        break  # Successfully generated a non-empty response
                    else:
                        print(f"Attempt {attempt + 1}: Model generated no new content. Retrying...")

                except Exception as e:
                    print(f"Error during generation (attempt {attempt + 1}): {str(e)}")
                    if attempt == max_retries - 1:
                        raise  # Re-raise the last exception if all retries failed

            if not response:
                raise ValueError("Model failed to generate new content after multiple attempts")

            # Add model response to conversation history
            self.conversation_history.append(f"AI: {response}")

            print(f"Final processed response: {response}")
            return response

        except ValueError as ve:
            error_message = f"Value error in chat method: {str(ve)}"
            print(f"ERROR: {error_message}")
            return self._handle_error("I'm sorry, but there was an issue with the input parameters or model output. Please try again.")

        except torch.cuda.OutOfMemoryError:
            error_message = "CUDA out of memory error. The input might be too long."
            print(f"ERROR: {error_message}")
            return self._handle_error("I'm sorry, but the input was too long for me to process. Can you try a shorter message?")

        except Exception as e:
            error_message = f"Unexpected error in chat method: {str(e)}"
            print(f"ERROR: {error_message}")
            if isinstance(e, torch.nn.modules.module.ModuleAttributeError):
                return self._handle_error("I'm experiencing some technical difficulties. Please try again later.")
            else:
                return self._handle_error("I encountered an unexpected error. Let's start over.")

    def _handle_error(self, message: str):
        """Helper method to handle errors in a consistent way."""
        self.conversation_history.append(f"AI: {message}")
        if len(self.conversation_history) > 10:
            print("Conversation history is getting long. Trimming to last 5 exchanges.")
            self.conversation_history = self.conversation_history[-10:]

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")

    def start_chat_session(self):
        print("Welcome to NextGenTorch Chat!")
        print("Type 'exit' to end the session or 'clear' to clear the conversation history.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                self.clear_history()
                continue
            response = self.chat(user_input)
            print(f"AI: {response}")

if __name__ == "__main__":
    chat_interface = ChatInterface()
    chat_interface.start_chat_session()
