# Create an __init__.py file for the tests package
init_content = '''"""
Test package for mayini deep learning framework.
"""
'''

# Save __init__.py
with open('test__init__.py', 'w') as f:
    f.write(init_content)

print("Created test __init__.py file")
