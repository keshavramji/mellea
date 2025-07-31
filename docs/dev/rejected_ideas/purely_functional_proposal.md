# Purely Functional Design Approach

## The Problem

Masataro observed that there is not a clean separation of concerns between backends and the rest of the code base. He suggested a purely functional approach instead.

## Proposed Solution

Below, I provide a mockup of Masataro's proposal in SML-esque syntax.

```
backend : str -> str | logit | prob

-- transformations on backends
cached_backend : cache * backend -> backend
lora_load : alora * backend -> backend
lora_add : alora * backend -> backend
lora_remove : alora * backend -> backend

-- getting into and out of the stdlib
prettyprinter : (CBlock | Component) list -> str list
parser : str | str list -> CBlock | Component
```

In addition to providing Masataro's desired separation of concerns, this approach has other beneifts:
1. Compelling to functional programmers
2. Less statefulness in backends
3. The user gets more direct control and explicit visibility over the variation on the backend they are using.

However, the approach also has some downsides:
1. We do not have time for another rewrite.
2. State needs to be threaded through the application by the end user. 
    - This is common in functional code-bases, and there isn't really a good solution despite decade(s?) of language design work around this question.
    - We could validate that this is a problem by doing a mockup and showing it to Kate or another early user.
3. The code is not pythonic. The code might be surprising to Python developers.


## Decision

We will not proceed with this approach because we do not have time for another rewrite. The pros and cons are documented and we can return to this topic later. Things that might trigger a review include subtantial pain, or an abundance of free time.