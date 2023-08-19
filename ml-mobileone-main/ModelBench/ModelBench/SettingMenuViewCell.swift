//
//  SettingMenuViewCell.swift
//
//  For licensing see accompanying LICENSE file.
//  Abstract:
//  The app's Setting menu cell object with a label and text field
//

import Foundation
import UIKit

class SettingMenuViewCell: UITableViewCell {
    lazy var backView: UIView = {
        let view = UIView(frame: CGRect(x: 0, y: 0, width: self.frame.width, height: 40))
        view.layer.backgroundColor = UIColor.clear.cgColor
        return view
    }()
    lazy var labelView: UILabel = {
        let view = UILabel(frame: CGRect(x: 5, y: 5, width: self.frame.width - 50, height: 30))
        return view
    }()
    lazy var valueView: UITextField = {
        let view = UITextField(frame: CGRect(x: 180, y: 5, width: 55, height: 30))
        view.keyboardType = .numbersAndPunctuation
        return view
    }()
    override func awakeFromNib() {
      super.awakeFromNib()
    }
    override func setSelected(_ selected: Bool, animated: Bool) {
        super.setSelected(selected, animated: animated)
        addSubview(backView)
        backView.addSubview(labelView)
        backView.addSubview(valueView)
    }
}
