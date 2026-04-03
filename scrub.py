import os
import re

TEXT_EXTENSIONS = {'.py', '.md', '.txt', '.gitignore'}
EXCLUDE_DIRS = {'.git', '__pycache__', 'venv', '.venv', 'embeddings', 'experiment_plots'}

REPLACEMENTS = [
    # Specific big targets
    (r'CENG442 - NLP Assignment Part 2', 'Azerbaijani Sentiment Analysis Pipeline'),
    (r'CENG442 - NLP Assignment', 'Azerbaijani Sentiment Analysis'),
    (r'CENG442\s*—\s*Natural Language Processing', 'Natural Language Processing Pipeline'),
    (r'CENG442\s*—\s*Azerbaijani Sentiment Analysis', 'Azerbaijani Sentiment Analysis'),
    (r'\*\*Mehmet Ali Yılmaz\*\* — Student ID: 21050111057\s*CENG442\s*—\s*Natural Language Processing\s*4th Year, 1st Semester', ''),
    (r'Mehmet Ali Yılmaz', ''),
    (r'Yılmaz', ''),
    (r'mehmet', ''),
    (r'21050111057', ''),
    (r'Student ID.*', ''),
    (r'4th Year.*', ''),
    (r'1st Semester', ''),
    
    # Gitignore specific scrub
    (r'CENG442\*', '*.pdf\n*.docx'),
    (r'CENG442_Assignment1_*.pdf', ''),
    (r'CENG442_Assignment1_Submission.txt', ''),
    
    # Generic phrasing
    (r'\b[Aa]ssignment\b', 'project'),
    (r'\b[Hh]omework\b', 'project'),
    (r'\b[Hh]w\b', 'project'),
    (r'Part 1', 'Phase 1'),
    (r'Part 2', 'Phase 2'),
    (r'process_assignment\.py', 'process_datasets.py'),
]

def scrub_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content
    for pattern, replacement in REPLACEMENTS:
        new_content = re.sub(pattern, replacement, new_content)

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Scrubbed: {filepath}")

def main():
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            # Skip the script itself
            if file == 'scrub.py':
                continue
                
            ext = os.path.splitext(file)[1]
            if ext in TEXT_EXTENSIONS or file == '.gitignore':
                filepath = os.path.join(root, file)
                try:
                    scrub_file(filepath)
                except Exception as e:
                    print(f"Failed {filepath}: {e}")

if __name__ == "__main__":
    main()
