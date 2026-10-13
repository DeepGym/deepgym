"""Environment authoring: from an idea to a QA-approved environment folder.

Modules:
    scaffold: ``deepgym new`` renders an env folder from the templates/ shipped here.
    check: ``deepgym check``, a fast loop over the folder contract, metadata schema,
        verifier source and protocol output, reference, empty and stub runs.
    approval: ``deepgym qa --approve`` writes approval.json (QA report plus content
        hashes); check flags an approval whose hashes no longer match.
"""
