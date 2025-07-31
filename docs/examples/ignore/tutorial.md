# Project M 

- [Chapter 1: Principles of Generative Programming](#chapter-1-principles-of-generative-programming)
- [Chapter 2: Getting Started with Generative Programming in M](#chapter-2-getting-started-with-generative-programming-in-m)
  - [Requirements](#requirements)
  - [Validating Requirements](#validating-requirements)
  - [Instruct - Validate - Repair](#instruct---validate---repair)
- [Chapter 3: Generative Programming Paradigms](#chapter-3-generative-programming-paradigms)
  - [Functional Programming](#functional-programming)
  - [Imperative Programming](#imperative-programming)
  - [Object-Oriented](#object-oriented)
- [Chapter 5: Working with Documents](#chapter-5-working-with-documents)
- [Chapter 6: Chats](#chapter-6-chats)
- [Chapter 7: Interoperability with Other Frameworks](#chapter-7-interoperability-with-other-frameworks)
- [Chapter 8: Scaling Generative Programs](#chapter-8-scaling-generative-programs)
- [Chapter 9: Prompt Engineering for `M`](#chapter-9-prompt-engineering-for-m)
  - [Custom  Templates](#custom--templates)


## Chapter 1: Principles of Generative Programming

1. Separate prompt engineering from control flow
2. Circumscribe nondeterminism and uncertainty.

## Chapter 2: Getting Started with Generative Programming in M

Let's get started with the first generative piece of code:

```python
import mellea

m = mellea.start_session(backend_name="ollama",
                         model_id=mellea.model_ids.IBM_GRANITE_3_3_8B)

email = m.instruct(
  "Write an email inviting interns to an office party at 3:30pm.")
```

We initialized a backend running Ollama on a local machine using the granite3.3-chat model. 
We then ask the model to generate an email and print it to the console.

> [!NOTE] 
> Mellea supports many other models and backends, but running this capable model on your own laptop is the default behavior, so from now on we will use `mellea.start_session()`.


Let's wrap this into a function with some arguments:

```python
import mellea


def write_email(m: mellea.MelleaSession, name: str, notes: str) -> str:
  email = m.instruct(
    "Write an email to {{name}} using the notes following: {{notes}}.",
    user_variables={"name": name, "notes": notes},
  )
  return email.value
```

Viola, we now have an email-writing function!

Notice how the instruct method can take a dictionary of variables as `user_variables`. These are filled by treading the instruction dedscription as a jinja template.

### Requirements

But how do we know that the generated email is a good one? We might have to define some 
**requirements** that this email has to fulfill to be "good" for our purposes:

```python
...
def write_email(name: str, notes: str) -> str:
    email =  m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        requirements=[
            "The email should have a salutation",
            "Use only lower-case letters"
        ],
        user_variables={"name": name, "notes": notes},
    )
    return email.value # str(email) also works.
```

We just added two requirements to the instruction which will be added to the model request.
But we don't check yet if these requirements are met at all. Let's add a **strategy** for 
validating the requirements:

```python
...
def write_email(name, notes) -> str:
    email_candidate =  m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        requirements=[
            "The email should have a salutation",
            "Use only lower-case letters"
        ],
        strategy=m.strategy.smc(budget=5), # see SMC: https://arxiv.org/pdf/2306.03081
        user_variables={"name": name, "notes": notes},
    )
    if email_candidate.success():
        return email_candidate.as_string()
    else:
        return email_candidate.candidates[0].as_string()
```

A couple of things happened here. First, we added a sampling `strategy` to the instruction.
This strategy (`m.strategy.smc()`) does (1) check if all requirements are met and (2a) if yes: return the generated email, or
(2b) if no: try generating a new emails with a stronger emphasize on the missing requirements. 
The maximum amount of five retries is given by the `budget=5` parameter.  

Sampling might not generate results that fulfill all requirements (`email_candidate.success()==False`). 
To handle this case we say that we simply return the first sampling result as final 
result using `email_candidate.candidates[0].as_string()`.

### Validating Requirements

Now that we defined requirements and sampling we should have a 
look into **how requirements are validated**. In our example above, 
the default validation strategy *LLM-as-a-judge* ([ref](https://arxiv.org/abs/2306.05685)) is used. 

Let's look on how we can customize requirement definitions:
```python
...
requirements=[
    req("The email should have a salutation", priority=m.prio.MUST), # == r1
    req("Do not respond in all-caps", priority=m.prio.SHOULD), # == r2
    req("Use only lower-case letters", validator=lambda x: x.as_string().lower() == x.as_string()), # == r3
    check("Do not mention purple elephants.") # == r4
 ],
...
```
Here, the first requirement (r1) has been given the highest priority of `MUST` which means, that it must be fulfilled
to lead to a positive result. In contrast, r2 should be fulfilled - i.e. if two samples have both fulfilled r1 but only 
one of them r2, then this one is the most preferred.

While r1 and r2 use LLM-as-a-judge for validation, the third requirement (r3) simply 
uses a function that takes the output of a sampling step and returns a boolean 
value indicating (un-)successful validation.    

The forth requirement is a `check()`. Checks are only used for validation, not for generation. 
Checks aim to avoid the "do not think about B" effect that often primes models (and humans) 
to do the opposite and "think" about B.

### Instruct - Validate - Repair

Now, we bring it all together into a first generative program using the **instruct-validate-repair** pattern: 

```python
import m
m.set_backend(m.OllamaBackend("ibm/granite3.2-chat"))

def write_email(name, notes) -> str:
    email_candidate =  m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        requirements=[
                m.req("The email should have a salutation", priority=m.prio.MUST), # == r1
                m.req("Do not respond in all-caps", priority=m.prio.SHOULD), # == r2
                m.req("Use only lower-case letters", validator=lambda x: x.as_string().lower() == x.as_string()), # == r3
                m.check("Do not mention purple elephants.") # == r4
        ],
        strategy=m.strategy.SMC(budget=5), 
        user_variables={"name": name, "notes": notes},
    )
    if email_candidate.success():
        return email_candidate.as_string()
    else:
        return email_candidate.candidates[0].as_string()
```

Btw, the `instruct()` method is a convenience function that wraps around an `m.Instruction()` Component,
`req()` wraps the `m.Requirement()` Component, etc. For the given example, this is the equivalent code
written by using these `m` Components:

```python
import m
from m import OllamaBackend, Instruction, Requirement
my_backend = OllamaBackend("ibm/granite3.2-chat")

def write_email(name, notes) -> str:
    email_instruction = Instruction(
            "Write an email to {{name}} using the notes following: {{notes}}.",
            requirements=[
                    Requirement("The email should have a salutation", priority=m.prio.MUST), # == r1
                    Requirement("Do not respond in all-caps", priority=m.prio.SHOULD), # == r2
                    Requirement("Use only lower-case letters", validator=lambda x: x.as_string().lower() == x.as_string()), # == r3
                    Requirement("Do not mention purple elephants.", validate_only=True) # == r4
            ],
            user_variables={"name": name, "notes": notes},
    )
    sampler = m.strategies.SMC(budget=5)
    email_candidate = sampler(email_instruction, backend = my_backend, validation_backend= my_backend)    

    if email_candidate.success():
        return email_candidate.as_string()
    else:
        return email_candidate.candidates[0].as_string()
```



--- TODO ---

Next we say that Instruct also provides an opinionated way of specifying relevant context.
and we show the subject line example where the email is the grounding context.

end of chapter.

## Chapter 3: Generative Programming Paradigms

*NATHAN*

In this chapter, we will continue building on the mail example from [chatper 2](#getting-started-with-generative-programming-in-m) explore how generative components of programs integrate into common programming paradigms.

### Functional Programming

this section is about parallelism

```python
import m
from my_orm import Customer

m.mify(Customer)

@dataclasses.dataclass
class CustomerInfo:
    name: str
    email: str
    notes: str

customer_summaries = m.pmap(
    lambda customer: customer.instruct("Summarize this customer blah blah here's the dict structure.", return_type=CustomerInfo, strategy=m.strats.smc_skip),
    Customer.select_all()
)

email_texts = m.pmap(
        lambda customer_info: write_email(customer_info.name, customer_info.notes),
        filter(lambda x: x is not None, customer_summaries)
)
```

Finally, let's send the emails by having the LLM write some code for us:

```python
def send_email(to_addr: str, to_name: str, subject: str, body: str, from_addr="nathan@nfulton.org", from_name="Nathan Fulton"):
    print("sending an email to...")

for email in emails:
    send_email(email.to, email.name, email.subject, email.body)
```


Suppose we actually wanted to send an email. We could write 


```python
@mify
def send_email(to_addr: str, to_name: str, subject: str, body: str, from_addr="nathan@nfulton.org", from_name="Nathan Fulton"):
    '''Send an email using gmail. The api key is in GMAIL_API_KEY env var.'''
    pass

for email in emails:
    send_email(email.to, email.name, email.subject, email.body)
```

### Imperative Programming

This section is about smaller-scale rejection sampling style programs.

We now maybe move beyond the email example, eg especially for example 2.

example 1: deeper dive into the instruct-validate-repair pattern.

example 2: react is an imperative program

example 3: exposing classical functions as tools (mify functions)

example 4: constrained decoding

### Object-Oriented

@mify on classes.




## Chapter 5: Working with Documents

*HENDRIK - 95%*

M makes it easy to work with documents. For that we provide generative wrappers
around [docling](https://github.com/docling-project/docling) documents.

Let's create a RichDocument from an arxiv paper:
```python
from m import RichDocument

rd = RichDocument.from_document_file("https://arxiv.org/pdf/1906.04043")
```
this loads the PDF file and parses it using the Docling parser into an 
intermediate representation.

You can now extract parts of the document, e.g. the figures...
```python
images = rd.get_images()
selected_image = images[2]
```

...and run a query method on it:
```python
vision_model = OllamaVisionLanguageModel()

description = selected_image.query("Describe the image in great detail.", backend=vision_model)
```
In this example, we use a VLM running locally on Ollama to generate a 
description for the image. This description can then be used, e.g., to store 
as metadata for the image in a retrieval database. 

We can also just search brute force search for a specific image:
```python
from m import RichImage
from typing import List

def search_for_images(question:str, images: List[RichImage]) -> List[RichImage]:
    results = [(im.query(question), im) for im in images]
    results = filter(lambda x:x[0].toLowerKey.startswWith("yes"), results)
    return [r[1] for r in results]

banana_images = search_for_images("Does the image contain a banana?", 
                                  rd.get_images())
```
The function iterates through the list of `RichImage` and checks if the 
answer to the question is "yes". 

We can apply the query function to another type of content. 
Here is an example for tables:
```python
sql_model = OllamaSQLModel()
table1 = rd.get_tables()[0]

median_auc = float(table1.query("What's the median AUC?", 
                    backend=sql_model))

normalized_table = table1.transform(f"Normalize AUC by using the median={median_auc}.", 
                                     backend=sql_model)
```
In the above example, we extract a table from the previously loaded document,
and ask the table for the median AUC value using the `query()` function as 
before. In the final step, we transform the table using the `transform()` function.
We also help the model by already providing the median value. 
For table query and transformation we use a local model that was trained on
table operations. We could have just used the general LLM as well, but with
probably subpar results.

Imagine the following two calls that should lead to the same result:
```python
table1_t = table1.transform("Transpose the table.")
table1_t2 = table1.transpose()
```
Every native function of `RichTable` is automatically registered
as a tool to the transform function. I.e., here the `.transform()` function calls the LLM and the LLM
will get back suggesting to use the very own `.transpose()` function to achieve the 
result - it will also give you a friendly warning that you could directly use the
function call instead of using the transform function. 



## Chapter 6: Chats

*NATHAN*

The `chat` function we saw in Chatper 2 can take *any* `m` Component.
This is particulartly useful for quick experimentation with more complex components such as the `RichDocument`:

```python
import m
from m.stdlib import MessageHistory, RichDocument, smc

def main():
    m.set_backend("ibm/granite3.2-chat") # TODO
    doc = RichDocument("myfile.pdf")
    messages: MessageHistory = m.chat(email_instruction)
    print(f"You had a chat with {len(messages)} messages.")
```

[[now what was that messagehistory thing? oh yeah if you wanna do chat bots we have some stdlib stuff for that.
here's how it works.]]

Note that using `m.chat` in this fashion is a useful tool for debugging and application development;
however, like Python's `input()`, it should not typically be used in most production code.

Many LLM applications organize a sequence of messages in terms of a "chat" dialog between a user and an assistant.
The `m` framework is opinionated away from chat-style interactions and toward generative programming, but
sometimes allows this classical mode of chat-style interaction is necessary.
The `m` framework supports this legacy mode of interaction via the `Message` and `MessageHistory` classes.

```python
import m
from m.stdlib import Message, MessageHistory

def main():
    m.set_backend("ibm/granite3.2-chat") # TODO 
    chat_history = MessageHistory()
    chat_history.add_message(UserMessage("Who is the queen of France?"))
    assistant_response: AssistantMessage = chat_history.generate_next_message()
    chat_history.add_message(UserMessage("Who is the king of the United States?"))
    assistant_response = chat_history.generate_next_message()
    print(chat_history)
```

Alternatively, you can start an interactive chat in a webchatui GUI by calling `m`'s built-in `chat` method on an empty `MessageHistory`.
[[When the user closes the browser tab with the webchatui instance, execution of the m program resumes.]]
You can think of this like Python's `input` functionality, but where the input modality is a chat GUI.
For example:

```python
import m
from m.stdlib import Message, MessageHistory

def main():
    m.set_backend("ibm/granite3.2-chat") # TODO 
    chat_history = MessageHistory()
    chat(chat_history)
    print(f"Your chat had {len(chat_history.get_messages())} messages")
```

## Chapter 7: Interoperability with Other Frameworks

*NATHAN*

TODO need to do some research Tomas should review.

All `m` programs are written in a `main()` function, which can be externalized using protocols such as the Agent Communication Protocol or Agent2Agent:

```python
from m.stdlib.agents import acp

@acp(name="Hello World m program")
def main():
    chat()
```

To register this with an ACP registry, run:

```
m agents register path/to/my/file.py
```

## Chapter 8: Scaling Generative Programs

*TBD*

special topic chapter on specifyting batching strategies, or something like that...

## Chapter 9: Prompt Engineering for `M`

*Nathan/Hen*


### Custom  Templates

don't bother writing this section until we actually write the code probably.

[[we should be more systrematic about where prompts get stored and refactor the PromptBasedBackends separated from the SpanBackends]]

TODO

```
    m/backends/prompts/
        stdlib/
            defaults/
                prompts.yaml
                formatter_helpers.py (optional)
            ibm/
                granite-defaults/
                    prompts.yaml
                    formatter_helpers.py (optional)
                granite-3.2.123.5.8/
                    prompts.yaml
                    formatter_helpers.py (optional)
                .../
            meta/
            ...
```


The `m` framework specifies reasonable opinionated defaults for how standard library primitives should be rendered.
Sometimes, you may want to override this default behavior.
To support this, every m `Component` has a `custom_template` keyword argument that can be used to...

```python
my_prompts.EMAIL_INSTRUCTION_PROMPT = "{{description}}\n\nAdhere to the following requirements: {% for r in requirements %} * {{r.description}}\n{% endif %}"

def write_email(name, notes) -> Optional[str]:
    email_text = instruct("Write an email to {{name}} using the notes in {{notes}}.",
            # note: rename to constraints?
             requirements=[
                req("The email should have a salutation", priority=m.HARD),
                req("Do not respond in all-caps", priority=m.MEDIUM),
                check("Do not mention purple elephants.")
             ],
             strategy=smc(budget=5),
             custom_template=my_prompts.EMAIL_INSTRUCTION_PROMPT)
```

You can also specify a list of possible templates in the `Strategy`. This will work for all stdlib `m` strategies,

```python
def write_email(name, notes) -> Optional[str]:
    email_text = instruct("Write an email to {{name}} using the notes in {{notes}}.",
            # note: rename to constraints?
             requirements=[
                req("The email should have a salutation", priority=m.HARD),
                req("Do not respond in all-caps", priority=m.MEDIUM),
                check("Do not mention purple elephants.")
             ],
             strategy=smc(budget=5, custom_templates=[my_prompts.EMAIL_INSTRUCTION_PROMPT, my_prmopts.ANOTHER_THING_HERE]) )
```

Note well: the strategy will be run, with the stated budget, for *each* of the specified templates. So, in this case, as many as 10 LLM calls may occur.
