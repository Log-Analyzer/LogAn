import pytest
from utils import replace_tags

def test_replace_script_tags():
    text = "<script>alert('test');</script>"
    expected = "alert('test');"
    assert replace_tags(text) == expected

def test_replace_br_tags():
    text = "Line1<br><br>Line2"
    expected = "Line1<br>Line2"
    assert replace_tags(text) == expected

def test_replace_other_tags():
    text = "<div>Content</div>"
    expected = "&lt;div&gt;Content&lt;/div&gt;"
    assert replace_tags(text) == expected

def test_mixed_tags():
    text = "<script>alert('test');</script><br><div>Content</div>"
    expected = "alert('test');<br>&lt;div&gt;Content&lt;/div&gt;"
    assert replace_tags(text) == expected

def test_no_tags():
    text = "No HTML tags"
    expected = "No HTML tags"
    assert replace_tags(text) == expected

def test_empty_string():
    text = ""
    expected = ""
    assert replace_tags(text) == expected

def test_nested_tags():
    text = "<div><span>Content</span></div>"
    expected = "&lt;div&gt;&lt;span&gt;Content&lt;/span&gt;&lt;/div&gt;"
    assert replace_tags(text) == expected

def test_self_closing_tags():
    text = "Image: <img src='image.jpg' />"
    expected = "Image: &lt;img src='image.jpg' /&gt;"
    assert replace_tags(text) == expected

def test_tags_with_attributes():
    text = "<a href='https://example.com'>Link</a>"
    expected = "&lt;a href='https://example.com'&gt;Link&lt;/a&gt;"
    assert replace_tags(text) == expected

def test_br_tag():
    text = "Line1<br>Line2"
    expected = "Line1<br>Line2"
    assert replace_tags(text) == expected

def test_self_closing_br_tag():
    text = "Line1<br/>Line2"
    expected = "Line1<br>Line2"  
    assert replace_tags(text) == expected

def test_multiple_br_tag():
    text = "Line1<br><br><br><br><br>Line2"
    expected = "Line1<br>Line2"
    assert replace_tags(text) == expected