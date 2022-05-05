import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15

Popup {
    id : _popup

    property var _config: {
        'panorama': null
    }

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
            Layout.minimumWidth : 450
            Layout.maximumWidth : 750
            spacing : 20

            Label {
                id : _title
                Layout.fillWidth : true

                font.pointSize : 16
                font.weight : Font.Medium

                text : '파노라마 생성·보정 설정'
            }

            ColumnLayout {
                spacing : 0
                // TODO ToolTip

                Label {
                    Layout.fillWidth : true

                    font.weight : Font.Medium
                    font.pointSize : 13

                    text : '파노라마 생성 설정'
                }

                GridLayout {
                    Layout.fillWidth : true
                    columns : 4
                    columnSpacing : 20 // TODO 두 column 사이 간격

                    Label {
                        Layout.fillWidth : true
                        text : '촬영 방법'
                    }
                    ComboBox {
                        id : _perspective
                        Layout.fillWidth : true

                        model : ['Panorama', 'Scan']
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '투영 방법'
                    }
                    ComboBox {
                        id : _warp
                        Layout.fillWidth : true

                        model : ['Plane', 'Spherical']
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '열화상 Blend'
                    }
                    ComboBox {
                        id : _ir_blend
                        Layout.fillWidth : true

                        model : ['Feather', 'Multiband', 'None']
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '실화상 Blend'
                    }
                    ComboBox {
                        id : _vis_blend
                        Layout.fillWidth : true

                        model : ['Feather', 'Multiband', 'None']
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '스케일'
                    }
                    FloatSpinBox {
                        id : _compose_scale
                        Layout.fillWidth : true

                        value : 100
                        from : 10
                        to : 100
                        stepSize : 5
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '변형 한계'
                    }
                    FloatSpinBox {
                        id : _warp_threshold
                        Layout.fillWidth : true

                        value : 2000
                        from : 100
                        to : 10000
                        stepSize : 100
                        decimals : 1
                    }

                }
            }

            ColumnLayout {
                GridLayout {
                    Layout.fillWidth : true
                    columns : 4
                    columnSpacing : 20

                    // title
                    Label {
                        Layout.columnSpan : 2

                        font.weight : Font.Medium
                        font.pointSize : 13

                        text : '열화상 전처리'
                    }

                    Label {
                        Layout.columnSpan : 2

                        font.weight : Font.Medium
                        font.pointSize : 13

                        text : '실화상 전처리'
                    }

                    // options
                    Label {
                        text : '명암 보정'
                    }
                    ComboBox {
                        id : _ir_contrast
                        Layout.fillWidth : true

                        model : ['Equalization', 'Normalization', 'None']
                    }

                    Label {
                        text : '명암 보정'
                    }
                    ComboBox {
                        id : _vis_contrast
                        Layout.fillWidth : true

                        model : ['Equalization', 'Normalization', 'None']
                    }
                    Label {
                        text : '노이즈 제거'
                    }
                    ComboBox {
                        id : _ir_denoise
                        Layout.fillWidth : true

                        model : ['Bilateral', 'Gaussian', 'None']
                    }

                    Label {
                        text : '노이즈 제거'
                    }
                    ComboBox {
                        id : _vis_denoise
                        Layout.fillWidth : true

                        model : ['Bilateral', 'Gaussian', 'None']
                    }
                }

                RowLayout {
                    Label {
                        text : '마스킹 온도'
                    }
                    FloatSpinBox {
                        id : _ir_masking_threshold
                        Layout.preferredWidth : _ir_denoise.width

                        value : -3000
                        from : -10000
                        to : 5000
                        stepSize : 100
                        decimals : 1
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

    function combo_value(value) {
        value = value.toLowerCase()
        if (value === 'none') {
            value = false
        }

        return value
    }

    function reset() {
        let st = _config['panorama']['stitch']

        _perspective.currentIndex = ['panorama', 'scan'].indexOf(st['perspective'])
        _warp.currentIndex = ['plane', 'spherical'].indexOf(st['warp'])
        _compose_scale.realValue = st['compose_scale']
        _warp_threshold.realValue = st['warp_threshold']

        let prep = _config['panorama']['preprocess']
        let contrast = ['equalization', 'normalization', null]
        let denoise = ['bilateral', 'gaussian', null]

        _ir_masking_threshold.realValue = prep['IR']['masking_threshold']
        _ir_contrast.currentIndex = contrast.indexOf(prep['IR']['contrast'])
        _ir_denoise.currentIndex = denoise.indexOf(prep['IR']['denoise'])

        _vis_contrast.currentIndex = contrast.indexOf(prep['VIS']['contrast'])
        _vis_denoise.currentIndex = denoise.indexOf(prep['VIS']['denoise'])
    }

    function configure() { // FIXME wd 설정 안 된 상태에서 오류 해결
        _config = {
            'panorama': {
                'stitch': {
                    'perspective': _perspective.currentText.toLowerCase(),
                    'warp': _warp.currentText.toLowerCase(),
                    'compose_scale': _compose_scale.realValue,
                    'warp_threshold': _warp_threshold.realValue,
                    'blend': {
                        'IR': combo_value(_ir_blend.currentText),
                        'VIS': combo_value(_vis_blend.currentText)
                    }
                },
                'preprocess': {
                    'IR': {
                        'contrast': combo_value(_ir_contrast.currentText),
                        'denoise': combo_value(_ir_denoise.currentText),
                        'masking_threshold': _ir_masking_threshold.realValue
                    },
                    'VIS': {
                        'contrast': combo_value(_vis_contrast.currentText),
                        'denoise': combo_value(_vis_denoise.currentText)
                    }
                }
            }
        }

        con.configure(JSON.stringify(_config))
    }

    function update_config(config) {
        _config['panorama'] = config['panorama']
        reset()
    }
}
