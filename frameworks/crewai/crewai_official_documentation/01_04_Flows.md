# Flows

> Learn how to create and manage AI workflows using CrewAI Flows.

## Overview

CrewAI Flows is a powerful feature designed to streamline the creation and management of AI workflows. Flows allow developers to combine and coordinate coding tasks and Crews efficiently, providing a robust framework for building sophisticated AI automations.

Flows allow you to create structured, event-driven workflows. They provide a seamless way to connect multiple tasks, manage state, and control the flow of execution in your AI applications. With Flows, you can easily design and implement multi-step processes that leverage the full potential of CrewAI's capabilities.

1. **Simplified Workflow Creation**: Easily chain together multiple Crews and tasks to create complex AI workflows.

2. **State Management**: Flows make it super easy to manage and share state between different tasks in your workflow.

3. **Event-Driven Architecture**: Built on an event-driven model, allowing for dynamic and responsive workflows.

4. **Flexible Control Flow**: Implement conditional logic, loops, and branching within your workflows.

## Getting Started

Let's create a simple Flow where you will use OpenAI to generate a random city in one task and then use that city to generate a fun fact in another task.

```python Code theme={null}

from crewai.flow.flow import Flow, listen, start
from dotenv import load_dotenv
from litellm import completion


class ExampleFlow(Flow):
    model = "gpt-4o-mini"

    @start()
    def generate_city(self):
        print("Starting flow")
        # Each flow state automatically gets a unique ID
        print(f"Flow State ID: {self.state['id']}")

        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "Return the name of a random city in the world.",
                },
            ],
        )

        random_city = response["choices"][0]["message"]["content"]
        # Store the city in our state
        self.state["city"] = random_city
        print(f"Random City: {random_city}")

        return random_city

    @listen(generate_city)
    def generate_fun_fact(self, random_city):
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"Tell me a fun fact about {random_city}",
                },
            ],
        )

        fun_fact = response["choices"][0]["message"]["content"]
        # Store the fun fact in our state
        self.state["fun_fact"] = fun_fact
        return fun_fact



flow = ExampleFlow()
flow.plot()
result = flow.kickoff()

print(f"Generated fun fact: {result}")
```

<img src="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-1.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=18b381277b7b017abf7cb19bc5e03923" alt="Flow Visual image" data-og-width="1913" width="1913" data-og-height="989" height="989" data-path="images/crewai-flow-1.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-1.png?w=280&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=78864d97e0fc7f225a5313c9fb650900 280w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-1.png?w=560&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=3d87938c680e7aa201798075fe19dcf8 560w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-1.png?w=840&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=36448790f7ca45e69ffdd3ceb2b2e713 840w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-1.png?w=1100&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=4d10a3f4f9ea1c9b0428fbb66f0fca17 1100w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-1.png?w=1650&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=928a75232235b73e9308d4d9cfeaf0e8 1650w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-1.png?w=2500&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=fa7022034285d0022ff07f97f6b675f7 2500w" />
In the above example, we have created a simple Flow that generates a random city using OpenAI and then generates a fun fact about that city. The Flow consists of two tasks: `generate_city` and `generate_fun_fact`. The `generate_city` task is the starting point of the Flow, and the `generate_fun_fact` task listens for the output of the `generate_city` task.

Each Flow instance automatically receives a unique identifier (UUID) in its state, which helps track and manage flow executions. The state can also store additional data (like the generated city and fun fact) that persists throughout the flow's execution.

When you run the Flow, it will:

1. Generate a unique ID for the flow state
2. Generate a random city and store it in the state
3. Generate a fun fact about that city and store it in the state
4. Print the results to the console

The state's unique ID and stored data can be useful for tracking flow executions and maintaining context between tasks.

**Note:** Ensure you have set up your `.env` file to store your `OPENAI_API_KEY`. This key is necessary for authenticating requests to the OpenAI API.

### @start()

The `@start()` decorator marks entry points for a Flow. You can:

* Declare multiple unconditional starts: `@start()`
* Gate a start on a prior method or router label: `@start("method_or_label")`
* Provide a callable condition to control when a start should fire

All satisfied `@start()` methods will execute (often in parallel) when the Flow begins or resumes.

### @listen()

The `@listen()` decorator is used to mark a method as a listener for the output of another task in the Flow. The method decorated with `@listen()` will be executed when the specified task emits an output. The method can access the output of the task it is listening to as an argument.

#### Usage

The `@listen()` decorator can be used in several ways:

1. **Listening to a Method by Name**: You can pass the name of the method you want to listen to as a string. When that method completes, the listener method will be triggered.

   ```python Code theme={null}
   @listen("generate_city")
   def generate_fun_fact(self, random_city):
       # Implementation
   ```

2. **Listening to a Method Directly**: You can pass the method itself. When that method completes, the listener method will be triggered.
   ```python Code theme={null}
   @listen(generate_city)
   def generate_fun_fact(self, random_city):
       # Implementation
   ```

### Flow Output

Accessing and handling the output of a Flow is essential for integrating your AI workflows into larger applications or systems. CrewAI Flows provide straightforward mechanisms to retrieve the final output, access intermediate results, and manage the overall state of your Flow.

#### Retrieving the Final Output

When you run a Flow, the final output is determined by the last method that completes. The `kickoff()` method returns the output of this final method.

Here's how you can access the final output:

<CodeGroup>
  ```python Code theme={null}
  from crewai.flow.flow import Flow, listen, start

  class OutputExampleFlow(Flow):
      @start()
      def first_method(self):
          return "Output from first_method"

      @listen(first_method)
      def second_method(self, first_output):
          return f"Second method received: {first_output}"


  flow = OutputExampleFlow()
  flow.plot("my_flow_plot")
  final_output = flow.kickoff()

  print("---- Final Output ----")
  print(final_output)
  ```

  ```text Output theme={null}
  ---- Final Output ----
  Second method received: Output from first_method
  ```
</CodeGroup>

<img src="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-2.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=3d987994d2c99a06a3cf149c71831fd5" alt="Flow Visual image" data-og-width="2015" width="2015" data-og-height="1040" height="1040" data-path="images/crewai-flow-2.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-2.png?w=280&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=e6b4e913cd2d4bf4dc67bdcb2e59cceb 280w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-2.png?w=560&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=245303e4f6e5bc30819aa9357561e7b3 560w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-2.png?w=840&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=32155410f336267e29c64407e22ae57e 840w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-2.png?w=1100&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=5dc414bc338e0475ae40aa3eedea0bd8 1100w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-2.png?w=1650&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=cfdf47937eb1f0a1f7e9ffdaab866e5a 1650w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-2.png?w=2500&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=ef7d71c39b8ea4ad865c514420df28d1 2500w" />

In this example, the `second_method` is the last method to complete, so its output will be the final output of the Flow.
The `kickoff()` method will return the final output, which is then printed to the console. The `plot()` method will generate the HTML file, which will help you understand the flow.

#### Accessing and Updating State

In addition to retrieving the final output, you can also access and update the state within your Flow. The state can be used to store and share data between different methods in the Flow. After the Flow has run, you can access the state to retrieve any information that was added or updated during the execution.

Here's an example of how to update and access the state:

<CodeGroup>
  ```python Code theme={null}
  from crewai.flow.flow import Flow, listen, start
  from pydantic import BaseModel

  class ExampleState(BaseModel):
      counter: int = 0
      message: str = ""

  class StateExampleFlow(Flow[ExampleState]):

      @start()
      def first_method(self):
          self.state.message = "Hello from first_method"
          self.state.counter += 1

      @listen(first_method)
      def second_method(self):
          self.state.message += " - updated by second_method"
          self.state.counter += 1
          return self.state.message

  flow = StateExampleFlow()
  flow.plot("my_flow_plot")
  final_output = flow.kickoff()
  print(f"Final Output: {final_output}")
  print("Final State:")
  print(flow.state)
  ```

  ```text Output theme={null}
  Final Output: Hello from first_method - updated by second_method
  Final State:
  counter=2 message='Hello from first_method - updated by second_method'
  ```
</CodeGroup>

<img src="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-2.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=3d987994d2c99a06a3cf149c71831fd5" alt="Flow Visual image" data-og-width="2015" width="2015" data-og-height="1040" height="1040" data-path="images/crewai-flow-2.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-2.png?w=280&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=e6b4e913cd2d4bf4dc67bdcb2e59cceb 280w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-2.png?w=560&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=245303e4f6e5bc30819aa9357561e7b3 560w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-2.png?w=840&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=32155410f336267e29c64407e22ae57e 840w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-2.png?w=1100&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=5dc414bc338e0475ae40aa3eedea0bd8 1100w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-2.png?w=1650&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=cfdf47937eb1f0a1f7e9ffdaab866e5a 1650w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-2.png?w=2500&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=ef7d71c39b8ea4ad865c514420df28d1 2500w" />

In this example, the state is updated by both `first_method` and `second_method`.
After the Flow has run, you can access the final state to see the updates made by these methods.

By ensuring that the final method's output is returned and providing access to the state, CrewAI Flows make it easy to integrate the results of your AI workflows into larger applications or systems,
while also maintaining and accessing the state throughout the Flow's execution.

## Flow State Management

Managing state effectively is crucial for building reliable and maintainable AI workflows. CrewAI Flows provides robust mechanisms for both unstructured and structured state management,
allowing developers to choose the approach that best fits their application's needs.

### Unstructured State Management

In unstructured state management, all state is stored in the `state` attribute of the `Flow` class.
This approach offers flexibility, enabling developers to add or modify state attributes on the fly without defining a strict schema.
Even with unstructured states, CrewAI Flows automatically generates and maintains a unique identifier (UUID) for each state instance.

```python Code theme={null}
from crewai.flow.flow import Flow, listen, start

class UnstructuredExampleFlow(Flow):

    @start()
    def first_method(self):
        # The state automatically includes an 'id' field
        print(f"State ID: {self.state['id']}")
        self.state['counter'] = 0
        self.state['message'] = "Hello from structured flow"

    @listen(first_method)
    def second_method(self):
        self.state['counter'] += 1
        self.state['message'] += " - updated"

    @listen(second_method)
    def third_method(self):
        self.state['counter'] += 1
        self.state['message'] += " - updated again"

        print(f"State after third_method: {self.state}")


flow = UnstructuredExampleFlow()
flow.plot("my_flow_plot")
flow.kickoff()
```

<img src="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-3.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=1d64a80a490430f29b7fa1085a3062c4" alt="Flow Visual image" data-og-width="1974" width="1974" data-og-height="1058" height="1058" data-path="images/crewai-flow-3.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-3.png?w=280&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=192f7a8605d3a5c12b6b61aa4a23917f 280w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-3.png?w=560&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=f41cbc9a268ba4bbb466fa2e2a1c2c1e 560w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-3.png?w=840&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=4d2315a6e69d8125e7e144f04180529f 840w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-3.png?w=1100&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=02eb5ffde3ef5936b2cf172160c72f72 1100w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-3.png?w=1650&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=3fc5bb51802a4a5d641834e19d24e565 1650w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-3.png?w=2500&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=85414106ada4d15dcb7bccc086194b84 2500w" />

**Note:** The `id` field is automatically generated and preserved throughout the flow's execution. You don't need to manage or set it manually, and it will be maintained even when updating the state with new data.

**Key Points:**

* **Flexibility:** You can dynamically add attributes to `self.state` without predefined constraints.
* **Simplicity:** Ideal for straightforward workflows where state structure is minimal or varies significantly.

### Structured State Management

Structured state management leverages predefined schemas to ensure consistency and type safety across the workflow.
By using models like Pydantic's `BaseModel`, developers can define the exact shape of the state, enabling better validation and auto-completion in development environments.

Each state in CrewAI Flows automatically receives a unique identifier (UUID) to help track and manage state instances. This ID is automatically generated and managed by the Flow system.

```python Code theme={null}
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel


class ExampleState(BaseModel):
    # Note: 'id' field is automatically added to all states
    counter: int = 0
    message: str = ""


class StructuredExampleFlow(Flow[ExampleState]):

    @start()
    def first_method(self):
        # Access the auto-generated ID if needed
        print(f"State ID: {self.state.id}")
        self.state.message = "Hello from structured flow"

    @listen(first_method)
    def second_method(self):
        self.state.counter += 1
        self.state.message += " - updated"

    @listen(second_method)
    def third_method(self):
        self.state.counter += 1
        self.state.message += " - updated again"

        print(f"State after third_method: {self.state}")


flow = StructuredExampleFlow()
flow.kickoff()
```

<img src="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-3.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=1d64a80a490430f29b7fa1085a3062c4" alt="Flow Visual image" data-og-width="1974" width="1974" data-og-height="1058" height="1058" data-path="images/crewai-flow-3.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-3.png?w=280&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=192f7a8605d3a5c12b6b61aa4a23917f 280w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-3.png?w=560&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=f41cbc9a268ba4bbb466fa2e2a1c2c1e 560w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-3.png?w=840&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=4d2315a6e69d8125e7e144f04180529f 840w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-3.png?w=1100&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=02eb5ffde3ef5936b2cf172160c72f72 1100w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-3.png?w=1650&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=3fc5bb51802a4a5d641834e19d24e565 1650w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-3.png?w=2500&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=85414106ada4d15dcb7bccc086194b84 2500w" />

**Key Points:**

* **Defined Schema:** `ExampleState` clearly outlines the state structure, enhancing code readability and maintainability.
* **Type Safety:** Leveraging Pydantic ensures that state attributes adhere to the specified types, reducing runtime errors.
* **Auto-Completion:** IDEs can provide better auto-completion and error checking based on the defined state model.

### Choosing Between Unstructured and Structured State Management

* **Use Unstructured State Management when:**

  * The workflow's state is simple or highly dynamic.
  * Flexibility is prioritized over strict state definitions.
  * Rapid prototyping is required without the overhead of defining schemas.

* **Use Structured State Management when:**
  * The workflow requires a well-defined and consistent state structure.
  * Type safety and validation are important for your application's reliability.
  * You want to leverage IDE features like auto-completion and type checking for better developer experience.

By providing both unstructured and structured state management options, CrewAI Flows empowers developers to build AI workflows that are both flexible and robust, catering to a wide range of application requirements.

## Flow Persistence

The @persist decorator enables automatic state persistence in CrewAI Flows, allowing you to maintain flow state across restarts or different workflow executions. This decorator can be applied at either the class level or method level, providing flexibility in how you manage state persistence.

### Class-Level Persistence

When applied at the class level, the @persist decorator automatically persists all flow method states:

```python  theme={null}
@persist  # Using SQLiteFlowPersistence by default
class MyFlow(Flow[MyState]):
    @start()
    def initialize_flow(self):
        # This method will automatically have its state persisted
        self.state.counter = 1
        print("Initialized flow. State ID:", self.state.id)

    @listen(initialize_flow)
    def next_step(self):
        # The state (including self.state.id) is automatically reloaded
        self.state.counter += 1
        print("Flow state is persisted. Counter:", self.state.counter)
```

### Method-Level Persistence

For more granular control, you can apply @persist to specific methods:

```python  theme={null}
class AnotherFlow(Flow[dict]):
    @persist  # Persists only this method's state
    @start()
    def begin(self):
        if "runs" not in self.state:
            self.state["runs"] = 0
        self.state["runs"] += 1
        print("Method-level persisted runs:", self.state["runs"])
```

### How It Works

1. **Unique State Identification**
   * Each flow state automatically receives a unique UUID
   * The ID is preserved across state updates and method calls
   * Supports both structured (Pydantic BaseModel) and unstructured (dictionary) states

2. **Default SQLite Backend**
   * SQLiteFlowPersistence is the default storage backend
   * States are automatically saved to a local SQLite database
   * Robust error handling ensures clear messages if database operations fail

3. **Error Handling**
   * Comprehensive error messages for database operations
   * Automatic state validation during save and load
   * Clear feedback when persistence operations encounter issues

### Important Considerations

* **State Types**: Both structured (Pydantic BaseModel) and unstructured (dictionary) states are supported
* **Automatic ID**: The `id` field is automatically added if not present
* **State Recovery**: Failed or restarted flows can automatically reload their previous state
* **Custom Implementation**: You can provide your own FlowPersistence implementation for specialized storage needs

### Technical Advantages

1. **Precise Control Through Low-Level Access**
   * Direct access to persistence operations for advanced use cases
   * Fine-grained control via method-level persistence decorators
   * Built-in state inspection and debugging capabilities
   * Full visibility into state changes and persistence operations

2. **Enhanced Reliability**
   * Automatic state recovery after system failures or restarts
   * Transaction-based state updates for data integrity
   * Comprehensive error handling with clear error messages
   * Robust validation during state save and load operations

3. **Extensible Architecture**
   * Customizable persistence backend through FlowPersistence interface
   * Support for specialized storage solutions beyond SQLite
   * Compatible with both structured (Pydantic) and unstructured (dict) states
   * Seamless integration with existing CrewAI flow patterns

The persistence system's architecture emphasizes technical precision and customization options, allowing developers to maintain full control over state management while benefiting from built-in reliability features.

## Flow Control

### Conditional Logic: `or`

The `or_` function in Flows allows you to listen to multiple methods and trigger the listener method when any of the specified methods emit an output.

<CodeGroup>
  ```python Code theme={null}
  from crewai.flow.flow import Flow, listen, or_, start

  class OrExampleFlow(Flow):

      @start()
      def start_method(self):
          return "Hello from the start method"

      @listen(start_method)
      def second_method(self):
          return "Hello from the second method"

      @listen(or_(start_method, second_method))
      def logger(self, result):
          print(f"Logger: {result}")



  flow = OrExampleFlow()
  flow.plot("my_flow_plot")
  flow.kickoff()
  ```

  ```text Output theme={null}
  Logger: Hello from the start method
  Logger: Hello from the second method
  ```
</CodeGroup>

<img src="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-4.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=88ce9c9f10781b835f170847bc541a13" alt="Flow Visual image" data-og-width="2026" width="2026" data-og-height="1016" height="1016" data-path="images/crewai-flow-4.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-4.png?w=280&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=796ce622251faa461b481eb5d7cdcf70 280w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-4.png?w=560&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=260fcd89a5b3a6a42a25dd4f41e7c5c6 560w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-4.png?w=840&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=b9268adb3abef93c7cce693a424a78ba 840w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-4.png?w=1100&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=75f3ad392bfd6b72bd29d701675899d6 1100w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-4.png?w=1650&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=dd771250338648e1f22c1463cb8e2ff0 1650w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-4.png?w=2500&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=b8e6fd63ec2ba23d9fa4f1dc2fd87143 2500w" />

When you run this Flow, the `logger` method will be triggered by the output of either the `start_method` or the `second_method`.
The `or_` function is used to listen to multiple methods and trigger the listener method when any of the specified methods emit an output.

### Conditional Logic: `and`

The `and_` function in Flows allows you to listen to multiple methods and trigger the listener method only when all the specified methods emit an output.

<CodeGroup>
  ```python Code theme={null}
  from crewai.flow.flow import Flow, and_, listen, start

  class AndExampleFlow(Flow):

      @start()
      def start_method(self):
          self.state["greeting"] = "Hello from the start method"

      @listen(start_method)
      def second_method(self):
          self.state["joke"] = "What do computers eat? Microchips."

      @listen(and_(start_method, second_method))
      def logger(self):
          print("---- Logger ----")
          print(self.state)

  flow = AndExampleFlow()
  flow.plot()
  flow.kickoff()
  ```

  ```text Output theme={null}
  ---- Logger ----
  {'greeting': 'Hello from the start method', 'joke': 'What do computers eat? Microchips.'}
  ```
</CodeGroup>

<img src="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-5.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=104318219be9d3502ac57ebb513aded7" alt="Flow Visual image" data-og-width="2062" width="2062" data-og-height="987" height="987" data-path="images/crewai-flow-5.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-5.png?w=280&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=6e9cb9d2b1ec2cb2aee2df008d3696c9 280w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-5.png?w=560&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=07cbcc6de6e8c8ae5da6c02a6fe4b457 560w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-5.png?w=840&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=afc6aad8f7276be4918527e553b5aa81 840w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-5.png?w=1100&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=e026919d9dff7ce0b0e592f4f2c0c4fd 1100w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-5.png?w=1650&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=03ea50c8681de2b8ea8cada6c0150e2c 1650w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-5.png?w=2500&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=61f9c7b0aceb69df6346ab2af321b779 2500w" />

When you run this Flow, the `logger` method will be triggered only when both the `start_method` and the `second_method` emit an output.
The `and_` function is used to listen to multiple methods and trigger the listener method only when all the specified methods emit an output.

### Router

The `@router()` decorator in Flows allows you to define conditional routing logic based on the output of a method.
You can specify different routes based on the output of the method, allowing you to control the flow of execution dynamically.

<CodeGroup>
  ```python Code theme={null}
  import random
  from crewai.flow.flow import Flow, listen, router, start
  from pydantic import BaseModel

  class ExampleState(BaseModel):
      success_flag: bool = False

  class RouterFlow(Flow[ExampleState]):

      @start()
      def start_method(self):
          print("Starting the structured flow")
          random_boolean = random.choice([True, False])
          self.state.success_flag = random_boolean

      @router(start_method)
      def second_method(self):
          if self.state.success_flag:
              return "success"
          else:
              return "failed"

      @listen("success")
      def third_method(self):
          print("Third method running")

      @listen("failed")
      def fourth_method(self):
          print("Fourth method running")


  flow = RouterFlow()
  flow.plot("my_flow_plot")
  flow.kickoff()
  ```

  ```text Output theme={null}
  Starting the structured flow
  Third method running
  Fourth method running
  ```
</CodeGroup>

<img src="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-6.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=f8cad73f073b4e936ef68d88545f1777" alt="Flow Visual image" data-og-width="1951" width="1951" data-og-height="1101" height="1101" data-path="images/crewai-flow-6.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-6.png?w=280&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=9a8462f42a9d9e14748d35312553ec6c 280w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-6.png?w=560&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=78e66eabae15099e2ef1d0c314d3cb04 560w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-6.png?w=840&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=a144377de810ed24f1d1aed1ba54d2d7 840w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-6.png?w=1100&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=2b17ebb2dd4eee4d086a8d0126a36c0d 1100w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-6.png?w=1650&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=4536b83aa7ff7e897f1193709ace944f 1650w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-6.png?w=2500&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=3b89cff817f7b4d05338f4a3a028f974 2500w" />

In the above example, the `start_method` generates a random boolean value and sets it in the state.
The `second_method` uses the `@router()` decorator to define conditional routing logic based on the value of the boolean.
If the boolean is `True`, the method returns `"success"`, and if it is `False`, the method returns `"failed"`.
The `third_method` and `fourth_method` listen to the output of the `second_method` and execute based on the returned value.

When you run this Flow, the output will change based on the random boolean value generated by the `start_method`.

## Adding Agents to Flows

Agents can be seamlessly integrated into your flows, providing a lightweight alternative to full Crews when you need simpler, focused task execution. Here's an example of how to use an Agent within a flow to perform market research:

```python  theme={null}
import asyncio
from typing import Any, Dict, List

from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.flow.flow import Flow, listen, start


# Define a structured output format
class MarketAnalysis(BaseModel):
    key_trends: List[str] = Field(description="List of identified market trends")
    market_size: str = Field(description="Estimated market size")
    competitors: List[str] = Field(description="Major competitors in the space")


# Define flow state
class MarketResearchState(BaseModel):
    product: str = ""
    analysis: MarketAnalysis | None = None


# Create a flow class
class MarketResearchFlow(Flow[MarketResearchState]):
    @start()
    def initialize_research(self) -> Dict[str, Any]:
        print(f"Starting market research for {self.state.product}")
        return {"product": self.state.product}

    @listen(initialize_research)
    async def analyze_market(self) -> Dict[str, Any]:
        # Create an Agent for market research
        analyst = Agent(
            role="Market Research Analyst",
            goal=f"Analyze the market for {self.state.product}",
            backstory="You are an experienced market analyst with expertise in "
            "identifying market trends and opportunities.",
            tools=[SerperDevTool()],
            verbose=True,
        )

        # Define the research query
        query = f"""
        Research the market for {self.state.product}. Include:
        1. Key market trends
        2. Market size
        3. Major competitors

        Format your response according to the specified structure.
        """

        # Execute the analysis with structured output format
        result = await analyst.kickoff_async(query, response_format=MarketAnalysis)
        if result.pydantic:
            print("result", result.pydantic)
        else:
            print("result", result)

        # Return the analysis to update the state
        return {"analysis": result.pydantic}

    @listen(analyze_market)
    def present_results(self, analysis) -> None:
        print("\nMarket Analysis Results")
        print("=====================")

        if isinstance(analysis, dict):
            # If we got a dict with 'analysis' key, extract the actual analysis object
            market_analysis = analysis.get("analysis")
        else:
            market_analysis = analysis

        if market_analysis and isinstance(market_analysis, MarketAnalysis):
            print("\nKey Market Trends:")
            for trend in market_analysis.key_trends:
                print(f"- {trend}")

            print(f"\nMarket Size: {market_analysis.market_size}")

            print("\nMajor Competitors:")
            for competitor in market_analysis.competitors:
                print(f"- {competitor}")
        else:
            print("No structured analysis data available.")
            print("Raw analysis:", analysis)


# Usage example
async def run_flow():
    flow = MarketResearchFlow()
    flow.plot("MarketResearchFlowPlot")
    result = await flow.kickoff_async(inputs={"product": "AI-powered chatbots"})
    return result


# Run the flow
if __name__ == "__main__":
    asyncio.run(run_flow())
```

<img src="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-7.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=6c60457e1a2b9bc0ef957c373a88359b" alt="Flow Visual image" data-og-width="1933" width="1933" data-og-height="959" height="959" data-path="images/crewai-flow-7.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-7.png?w=280&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=4e6a743b2b19cd86dadbbd015d0a0393 280w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-7.png?w=560&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=740f254bb03d60cd011911dab702ca77 560w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-7.png?w=840&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=b93c5bde69019cdc34c143bcc0885743 840w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-7.png?w=1100&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=4eabe8d3536d6588a14157b60bc7a1e0 1100w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-7.png?w=1650&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=e76b5df7821722a59d3267f3a0eff3ed 1650w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-7.png?w=2500&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=3a9775e3f5798ccb73e5feb7e53319fd 2500w" />

This example demonstrates several key features of using Agents in flows:

1. **Structured Output**: Using Pydantic models to define the expected output format (`MarketAnalysis`) ensures type safety and structured data throughout the flow.

2. **State Management**: The flow state (`MarketResearchState`) maintains context between steps and stores both inputs and outputs.

3. **Tool Integration**: Agents can use tools (like `WebsiteSearchTool`) to enhance their capabilities.

## Adding Crews to Flows

Creating a flow with multiple crews in CrewAI is straightforward.

You can generate a new CrewAI project that includes all the scaffolding needed to create a flow with multiple crews by running the following command:

```bash  theme={null}
crewai create flow name_of_flow
```

This command will generate a new CrewAI project with the necessary folder structure. The generated project includes a prebuilt crew called `poem_crew` that is already working. You can use this crew as a template by copying, pasting, and editing it to create other crews.

### Folder Structure

After running the `crewai create flow name_of_flow` command, you will see a folder structure similar to the following:

| Directory/File         | Description                                                         |
| :--------------------- | :------------------------------------------------------------------ |
| `name_of_flow/`        | Root directory for the flow.                                        |
| ├── `crews/`           | Contains directories for specific crews.                            |
| │ └── `poem_crew/`     | Directory for the "poem\_crew" with its configurations and scripts. |
| │ ├── `config/`        | Configuration files directory for the "poem\_crew".                 |
| │ │ ├── `agents.yaml`  | YAML file defining the agents for "poem\_crew".                     |
| │ │ └── `tasks.yaml`   | YAML file defining the tasks for "poem\_crew".                      |
| │ ├── `poem_crew.py`   | Script for "poem\_crew" functionality.                              |
| ├── `tools/`           | Directory for additional tools used in the flow.                    |
| │ └── `custom_tool.py` | Custom tool implementation.                                         |
| ├── `main.py`          | Main script for running the flow.                                   |
| ├── `README.md`        | Project description and instructions.                               |
| ├── `pyproject.toml`   | Configuration file for project dependencies and settings.           |
| └── `.gitignore`       | Specifies files and directories to ignore in version control.       |

### Building Your Crews

In the `crews` folder, you can define multiple crews. Each crew will have its own folder containing configuration files and the crew definition file. For example, the `poem_crew` folder contains:

* `config/agents.yaml`: Defines the agents for the crew.
* `config/tasks.yaml`: Defines the tasks for the crew.
* `poem_crew.py`: Contains the crew definition, including agents, tasks, and the crew itself.

You can copy, paste, and edit the `poem_crew` to create other crews.

### Connecting Crews in `main.py`

The `main.py` file is where you create your flow and connect the crews together. You can define your flow by using the `Flow` class and the decorators `@start` and `@listen` to specify the flow of execution.

Here's an example of how you can connect the `poem_crew` in the `main.py` file:

```python Code theme={null}
#!/usr/bin/env python
from random import randint

from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from .crews.poem_crew.poem_crew import PoemCrew

class PoemState(BaseModel):
    sentence_count: int = 1
    poem: str = ""

class PoemFlow(Flow[PoemState]):

    @start()
    def generate_sentence_count(self):
        print("Generating sentence count")
        self.state.sentence_count = randint(1, 5)

    @listen(generate_sentence_count)
    def generate_poem(self):
        print("Generating poem")
        result = PoemCrew().crew().kickoff(inputs={"sentence_count": self.state.sentence_count})

        print("Poem generated", result.raw)
        self.state.poem = result.raw

    @listen(generate_poem)
    def save_poem(self):
        print("Saving poem")
        with open("poem.txt", "w") as f:
            f.write(self.state.poem)

def kickoff():
    poem_flow = PoemFlow()
    poem_flow.kickoff()


def plot():
    poem_flow = PoemFlow()
    poem_flow.plot("PoemFlowPlot")

if __name__ == "__main__":
    kickoff()
    plot()
```

In this example, the `PoemFlow` class defines a flow that generates a sentence count, uses the `PoemCrew` to generate a poem, and then saves the poem to a file. The flow is kicked off by calling the `kickoff()` method. The PoemFlowPlot will be generated by `plot()` method.

<img src="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-8.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=5321ca5d1f3c662dc7cff6950ba46000" alt="Flow Visual image" data-og-width="1901" width="1901" data-og-height="1032" height="1032" data-path="images/crewai-flow-8.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-8.png?w=280&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=5179beeb8c5b02eafdc1fce722004529 280w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-8.png?w=560&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=994b487041be812c1df343b23b5da9f2 560w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-8.png?w=840&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=244859a74e398490fa313beb91a3b9a7 840w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-8.png?w=1100&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=7ab59d8b176f60f2ba882eca41100ce1 1100w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-8.png?w=1650&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=c817c66b2d016c0b1bc203c32413d08e 1650w, https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crewai-flow-8.png?w=2500&fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=f5936902cb8e9e405a65ab3e1adbbd43 2500w" />

### Running the Flow

(Optional) Before running the flow, you can install the dependencies by running:

```bash  theme={null}
crewai install
```

Once all of the dependencies are installed, you need to activate the virtual environment by running:

```bash  theme={null}
source .venv/bin/activate
```

After activating the virtual environment, you can run the flow by executing one of the following commands:

```bash  theme={null}
crewai flow kickoff
```

or

```bash  theme={null}
uv run kickoff
```

The flow will execute, and you should see the output in the console.

## Plot Flows

Visualizing your AI workflows can provide valuable insights into the structure and execution paths of your flows. CrewAI offers a powerful visualization tool that allows you to generate interactive plots of your flows, making it easier to understand and optimize your AI workflows.

### What are Plots?

Plots in CrewAI are graphical representations of your AI workflows. They display the various tasks, their connections, and the flow of data between them. This visualization helps in understanding the sequence of operations, identifying bottlenecks, and ensuring that the workflow logic aligns with your expectations.

### How to Generate a Plot

CrewAI provides two convenient methods to generate plots of your flows:

#### Option 1: Using the `plot()` Method

If you are working directly with a flow instance, you can generate a plot by calling the `plot()` method on your flow object. This method will create an HTML file containing the interactive plot of your flow.

```python Code theme={null}
# Assuming you have a flow instance
flow.plot("my_flow_plot")
```

This will generate a file named `my_flow_plot.html` in your current directory. You can open this file in a web browser to view the interactive plot.

#### Option 2: Using the Command Line

If you are working within a structured CrewAI project, you can generate a plot using the command line. This is particularly useful for larger projects where you want to visualize the entire flow setup.

```bash  theme={null}
crewai flow plot
```

This command will generate an HTML file with the plot of your flow, similar to the `plot()` method. The file will be saved in your project directory, and you can open it in a web browser to explore the flow.

### Understanding the Plot

The generated plot will display nodes representing the tasks in your flow, with directed edges indicating the flow of execution. The plot is interactive, allowing you to zoom in and out, and hover over nodes to see additional details.

By visualizing your flows, you can gain a clearer understanding of the workflow's structure, making it easier to debug, optimize, and communicate your AI processes to others.

### Conclusion

Plotting your flows is a powerful feature of CrewAI that enhances your ability to design and manage complex AI workflows. Whether you choose to use the `plot()` method or the command line, generating plots will provide you with a visual representation of your workflows, aiding in both development and presentation.

## Next Steps

If you're interested in exploring additional examples of flows, we have a variety of recommendations in our examples repository. Here are four specific flow examples, each showcasing unique use cases to help you match your current problem type to a specific example:

1. **Email Auto Responder Flow**: This example demonstrates an infinite loop where a background job continually runs to automate email responses. It's a great use case for tasks that need to be performed repeatedly without manual intervention. [View Example](https://github.com/crewAIInc/crewAI-examples/tree/main/email_auto_responder_flow)

2. **Lead Score Flow**: This flow showcases adding human-in-the-loop feedback and handling different conditional branches using the router. It's an excellent example of how to incorporate dynamic decision-making and human oversight into your workflows. [View Example](https://github.com/crewAIInc/crewAI-examples/tree/main/lead-score-flow)

3. **Write a Book Flow**: This example excels at chaining multiple crews together, where the output of one crew is used by another. Specifically, one crew outlines an entire book, and another crew generates chapters based on the outline. Eventually, everything is connected to produce a complete book. This flow is perfect for complex, multi-step processes that require coordination between different tasks. [View Example](https://github.com/crewAIInc/crewAI-examples/tree/main/write_a_book_with_flows)

4. **Meeting Assistant Flow**: This flow demonstrates how to broadcast one event to trigger multiple follow-up actions. For instance, after a meeting is completed, the flow can update a Trello board, send a Slack message, and save the results. It's a great example of handling multiple outcomes from a single event, making it ideal for comprehensive task management and notification systems. [View Example](https://github.com/crewAIInc/crewAI-examples/tree/main/meeting_assistant_flow)

By exploring these examples, you can gain insights into how to leverage CrewAI Flows for various use cases, from automating repetitive tasks to managing complex, multi-step processes with dynamic decision-making and human feedback.

Also, check out our YouTube video on how to use flows in CrewAI below!

<iframe className="w-full aspect-video rounded-xl" src="https://www.youtube.com/embed/MTb5my6VOT8" title="CrewAI Flows overview" frameBorder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerPolicy="strict-origin-when-cross-origin" allowFullScreen />

## Running Flows

There are two ways to run a flow:

### Using the Flow API

You can run a flow programmatically by creating an instance of your flow class and calling the `kickoff()` method:

```python  theme={null}
flow = ExampleFlow()
result = flow.kickoff()
```

### Using the CLI

Starting from version 0.103.0, you can run flows using the `crewai run` command:

```shell  theme={null}
crewai run
```

This command automatically detects if your project is a flow (based on the `type = "flow"` setting in your pyproject.toml) and runs it accordingly. This is the recommended way to run flows from the command line.

For backward compatibility, you can also use:

```shell  theme={null}
crewai flow kickoff
```

However, the `crewai run` command is now the preferred method as it works for both crews and flows.
