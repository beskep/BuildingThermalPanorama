import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import "../Button"

RowLayout {
    property int index: -1
    property var skip: [4, 1] // !separate, separate

    ToolButton {
        text: '이전'
        icon: '\ueac3'
        enabled: index !== -1 & index > 0
        onReleased: {
            let prev = index - 1;
            if (prev === skip[+app.separate_panorama])
                prev -= 1;

            app.set_panel(prev);
        }
    }

    ToolButton {
        text: '다음'
        icon2: '\ueac9'
        enabled: index !== -1 & index < app.panel_count - 1
        onReleased: {
            let next = index + 1;
            if (next === skip[+app.separate_panorama])
                next += 1;

            app.set_panel(next);
        }
    }

}
