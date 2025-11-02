from string import Template

prompt_template = Template("""
You are a ${role}.

 **Goal**: ${goal}

 **Guide**:
${guide}

 **Example**:
${example}

Now, based on the above, generate your response below.
""")
