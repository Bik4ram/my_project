from semrag.reasoning.coding.intent import classify_intent, is_create_request, request_target_noun
from semrag.reasoning.coding.models import CodingIntent


def test_information_trace():
    assert classify_intent("Trace the I2C READ transaction.") is CodingIntent.INFORMATION


def test_information_explain():
    assert classify_intent("Explain how the design works") is CodingIntent.INFORMATION


def test_information_interrogative():
    assert classify_intent("What is the purpose of the sequencer?") is CodingIntent.INFORMATION


def test_code_change_create_test():
    assert classify_intent("Create a READ test following the existing i2c_master UVM test pattern") is CodingIntent.CODE_CHANGE


def test_code_change_imperative():
    assert classify_intent("Add a new driver class named i2c_master_read_driver") is CodingIntent.CODE_CHANGE


def test_code_change_modify():
    assert classify_intent("Modify i2c_master_driver to add a timeout") is CodingIntent.CODE_CHANGE


def test_debug_fix_lint():
    assert classify_intent("Fix the failing lint in i2c_master_top") is CodingIntent.DEBUG


def test_debug_syntax_error():
    assert classify_intent("The new file does not compile, fix it") is CodingIntent.DEBUG


def test_debug_bug():
    assert classify_intent("Debug the incorrect read behaviour") is CodingIntent.DEBUG


def test_review():
    assert classify_intent("Review the i2c_master_driver code for correctness") is CodingIntent.REVIEW


def test_review_check():
    assert classify_intent("Check the correctness of the testbench") is CodingIntent.REVIEW


def test_ambiguous_greeting():
    assert classify_intent("Hello there") is CodingIntent.AMBIGUOUS


def test_ambiguous_short():
    assert classify_intent("") is CodingIntent.AMBIGUOUS


def test_ambiguous_no_verb():
    assert classify_intent("the driver file") is CodingIntent.AMBIGUOUS


def test_question_with_change_verb_is_change():
    # "Can you add ...?" is a question but clearly requests a change.
    assert classify_intent("Can you add a READ test to the project?") is CodingIntent.CODE_CHANGE


def test_is_create_request_true():
    assert is_create_request("Create a READ test following the existing i2c_master UVM test pattern")
    assert is_create_request("Write a new sequence file for the master")
    assert is_create_request("Add a new test file for the write path")


def test_is_create_request_false_for_modify():
    assert not is_create_request("Add a constructor to the existing test")
    assert not is_create_request("Update the test class name")
    assert not is_create_request("Name the endmodule in i2c.sv")
    assert not is_create_request("Trace the I2C READ transaction.")


def test_is_create_request_false_for_debug():
    assert not is_create_request("Fix the failing lint in i2c_master_top")


def test_request_target_noun():
    assert request_target_noun("Create a READ test") == "read_test"
    assert request_target_noun("Write a new write sequence") == "write_sequence"
    assert request_target_noun("Update the test class name") is None