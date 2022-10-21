from src.image_recognition.board_reader import BoardReader


def test_board_reader_extract_blocks():
    test_file = "test/image_recognition/photo_test.jpeg"

    board_reader = BoardReader(test_file)
    blocks = board_reader.extract_blocks_from_image()

    assert len(blocks) == 81


def test_extracting_blocks_prepares_the_image_for_processing(mocker):
    test_file = "test/image_recognition/photo_test.jpeg"

    board_reader = BoardReader(test_file)
    prepare_image_spy = mocker.spy(board_reader, "prepare_image")
    blocks = board_reader.extract_blocks_from_image()

    prepare_image_spy.assert_called_once()


def test_extracting_blocks_identifies_image_contours(mocker):
    test_file = "test/image_recognition/photo_test.jpeg"

    board_reader = BoardReader(test_file)
    contours_spy = mocker.spy(board_reader, "identify_contours")
    blocks = board_reader.extract_blocks_from_image()

    contours_spy.assert_called_once()
    board_reader.contours != None


def test_extracting_blocks_adjusts_image_perspective(mocker):
    test_file = "test/image_recognition/photo_test.jpeg"

    board_reader = BoardReader(test_file)
    perspective_adjust_spy = mocker.spy(board_reader, "adjust_for_perspective")
    blocks = board_reader.extract_blocks_from_image()

    perspective_adjust_spy.assert_called_once()
    board_reader.board != None
    board_reader.location != None
