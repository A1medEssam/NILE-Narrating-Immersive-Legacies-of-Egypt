from Ramesses_rag_stone import preprocess_text

#some dummy test cases from my pov...
test_cases = [
    ("a) - .- RAMESSES JI", "RAMESSES JI"),
    ("tel | 7", ""),
    ("{ SL }", ""),
    ("Chapter 7-THE CORRIDORS", "Chapter 7-THE CORRIDORS"),
    ("=< ae K. A. Kitchen", "ae K. A. Kitchen"),
    ("word-\nbroken", "wordbroken"),
    ("See (Figure 3)", "See"),
    ("Ramesses II (see fig.5)", "Ramesses II"),
    ("Section 1.2\n\nContent", "Section 1.2 Content"),
    ("Dr. A. B. Smith", "Dr. A. B. Smith"),
    ("Page 23 of 284", "Page 23 of 284")
]

all_passed = True
for i, (original, expected) in enumerate(test_cases, 1):
    result = preprocess_text(original)
    passed = result == expected
    all_passed = all_passed and passed
    
    print(f"Test #{i}: {'PASS' if passed else 'FAIL'}")
    print(f"Original: {original!r}")
    print(f"Cleaned: {result!r}")
    if not passed:
        print(f"Expected: {expected!r}")
    print("-" * 50)

print("\nFinal Result:", "ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED")