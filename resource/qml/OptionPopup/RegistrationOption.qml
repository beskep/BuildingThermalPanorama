import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15

Popup {
    id : _popup

    property var _config: {'registration':null}

    anchors.centerIn : Overlay.overlay
    Material.elevation : 5
    padding : 0
    height : _content.implicitHeight

    ColumnLayout {
        id : _content

        anchors.fill : parent

        ColumnLayout {
            Layout.fillWidth : true
            Layout.fillHeight : true
            Layout.margins : 20
            Layout.minimumWidth : 400
            Layout.maximumWidth : 750
            spacing : 20

            Label {
                id : _title
                Layout.fillWidth : true

                font.pointSize : 16
                font.weight : Font.Medium

                text : '자동 열·실화상 정합 설정'
            }

            ColumnLayout {
                spacing : 0

                Label {
                    Layout.fillWidth : true

                    font.weight : Font.Medium
                    font.pointSize : 13

                    text : '전처리 설정'
                }

                RowLayout {
                    Label {
                        Layout.fillWidth : true
                        text : 'HistogramEq'
                    }
                    CheckBox {
                        id : _hist_eq
                        checked : true
                        ToolTip.visible : hovered
                        ToolTip.text : '히스토그램 균일화를 통해 명암을 보정'
                    }
                    Rectangle {
                        width : 50
                    }
                    Label {
                        Layout.fillWidth : true
                        text : 'Sharpening'
                    }
                    CheckBox {
                        id : _unsharp
                        ToolTip.visible : hovered
                        ToolTip.text : '물체 간 경계선의 명암 차이를 강화'
                    }
                }
            }

            ColumnLayout {
                spacing : 0

                Label {
                    Layout.fillWidth : true

                    font.weight : Font.Medium
                    font.pointSize : 13

                    text : '수치적 정합 설정'
                }

                GridLayout {
                    columns : 2
                    Layout.fillWidth : true

                    Label {
                        text : '최적화 변수'
                        Layout.fillWidth : true
                    }
                    ComboBox {
                        id : _metric
                        Layout.fillWidth : true

                        model : ['JointHistMI', 'MattesMI', 'MeanSquare']
                    }
                    Label {
                        text : '영상 변환 방법'
                        Layout.fillWidth : true
                    }
                    ComboBox {
                        id : _transformation
                        Layout.fillWidth : true

                        model : ['Similarity', 'Affine']
                    }
                    Label {
                        text : '밝기 구간 분할 방법'
                        Layout.fillWidth : true
                    }
                    ComboBox {
                        id : _bins
                        Layout.fillWidth : true

                        model : ['Auto', 'Freedman-Diaconis', 'Square Root']
                    }
                    Label {
                        text : '최적화 방법'
                        Layout.fillWidth : true
                    }
                    ComboBox {
                        id : _optimizer
                        Layout.fillWidth : true

                        model : ['Gradient Descent', 'Powell']
                    }
                }
            }

            RowLayout {
                Layout.alignment : Qt.AlignRight | Qt.AlignBottom
                Button {
                    flat : true
                    text : 'Cancel'
                    onClicked : {
                        reset();
                        _popup.close()
                    }
                }

                Button {
                    flat : true
                    text : 'OK'
                    onClicked : {
                        configure();
                        _popup.close();
                    }
                }
            }
        }
    }

    function reset() {
        let cfg = _config['registration']
        if (! cfg) {
            return
        }

        _hist_eq.checked = cfg['preprocess']['equalize_histogram'];
        _unsharp.checked = cfg['preprocess']['unsharp'];
        _metric.currentIndex = ['JointHistMI', 'MattesMI', 'MeanSquare'].indexOf(cfg['metric']);
        _transformation.currentIndex = ['Similarity', 'Affine'].indexOf(cfg['transformation']);
        _bins.currentIndex = ['auto', 'fd', 'sqrt'].indexOf(cfg['bins']);
        _optimizer.currentIndex = ['gradient_descent', 'powell'].indexOf(cfg['optimizer']);
    }

    function configure() {
        _config = {
            'registration': {
                'preprocess': {
                    'equalize_histogram': (_hist_eq.checkState === Qt.Checked),
                    'unsharp': (_unsharp.checkState === Qt.Checked)
                },
                'metric': _metric.currentText,
                'transformation': _transformation.currentText,
                'bins': ['auto', 'fd', 'sqrt'][_bins.currentIndex],
                'optimizer': ['gradient_descent', 'powell'][_optimizer.currentIndex]
            }
        }

        con.configure(JSON.stringify(_config))
    }

    function update_config(config) {
        _config['registration'] = config['registration']
        reset()
    }
}
