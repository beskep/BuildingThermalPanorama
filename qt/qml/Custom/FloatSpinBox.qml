import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15

SpinBox {
    id: spinbox

    property int decimals: 2
    property real realValue: value / 100

    editable: true
    wheelEnabled: true
    from: 0
    value: 100
    to: 10000
    stepSize: 100
    textFromValue: function(value, locale) {
        return Number(value / 100).toLocaleString(locale, 'f', spinbox.decimals);
    }
    valueFromText: function(text, locale) {
        return Number.fromLocaleString(locale, text) * 100;
    }

    validator: DoubleValidator {
        bottom: Math.min(spinbox.from, spinbox.to)
        top: Math.max(spinbox.from, spinbox.to)
    }

}
