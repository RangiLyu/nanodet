//
//  ViewController.swift
//  ModelBench
//
//  For licensing see accompanying LICENSE file.
//  Abstract:
//  The app's primary view controller that presents the benchmark interface.
//

import UIKit
import CoreML

let inferenceQueue = DispatchQueue(label: "inferenceQueue")

class ViewController: UIViewController, UITableViewDelegate, UITableViewDataSource, UITextFieldDelegate {
    // Benchmark configuration and data
    var rounds: Int = 20
    var inferencesPerRound: Int = 50
    let warmUps: Int = 1
    var trims: Int = 10
    var running: Bool = false
    var averageLatency: Double = 0.0
    var latencyLow: Double = .infinity
    var latencyHigh: Double = 0.0
    var latencies: [Double] = []
    var roundsLatencies: [[Double]] = []
    var averagies: [Double] = []
    var averageAll: Double = .infinity

    // All MobileOne models
    lazy var modelMobileOneS0: mobileone_s0 = try! mobileone_s0(configuration: MLModelConfiguration())
    lazy var modelMobileOneS1: mobileone_s1 = try! mobileone_s1(configuration: MLModelConfiguration())
    lazy var modelMobileOneS2: mobileone_s2 = try! mobileone_s2(configuration: MLModelConfiguration())
    lazy var modelMobileOneS3: mobileone_s3 = try! mobileone_s3(configuration: MLModelConfiguration())
    lazy var modelMobileOneS4: mobileone_s4 = try! mobileone_s4(configuration: MLModelConfiguration())

    // menu data source
    var settingMenuViewDataSource: [(String, Int, Int)]!
    var settingMenuView = UITableView()

    var modelSelection: ModelType = .mobileOneS0

    @IBOutlet weak var averageAllLatencyField: UITextField!
    @IBOutlet weak var averageLastLatencyField: UITextField!
    @IBOutlet weak var latencyLowField: UITextField!
    @IBOutlet weak var latencyHighField: UITextField!

    @IBOutlet weak var runButton: UIButton!
    @IBOutlet weak var latenciesTextView: UITextView!
    @IBOutlet weak var settingButton: UIButton!
    @IBOutlet weak var modelPopupButton: UIButton!

    enum ModelType: String, CaseIterable {
        case mobileOneS0 = "MobileOne-S0"
        case mobileOneS1 = "MobileOne-S1"
        case mobileOneS2 = "MobileOne-S2"
        case mobileOneS3 = "MobileOne-S3"
        case mobileOneS4 = "MobileOne-S4"
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
        latenciesTextView.isEditable = false

        setupSettingMenuView()
        settingMenuView.isHidden = true

        // Set up setting popup menu
        modelPopupButton.menu = UIMenu(title: "", children: (
            0..<ModelType.allCases.count).reversed().map {
                UIAction(title: "\(ModelType.allCases[$0].rawValue)", handler: menuModelHandler)
            })
        modelPopupButton.menu?.children.forEach { action in
            if let action = action as? UIAction {
                if action.title == self.modelSelection.rawValue {
                    action.state = .on
                }
            }
        }
    }

    @IBAction func RunButtonClicked(_ sender: Any) {
        self.roundsLatencies = []
        self.runButton.isEnabled = false
        self.averageAllLatencyField.text = ""
        self.averageLastLatencyField.text = ""
        self.latencyLowField.text = ""
        self.latencyHighField.text = ""
        self.latenciesTextView.text = "Running...\n"

        self.averagies.removeAll()
        self.latencyLow = .infinity
        self.latencyHigh = 0.0
        self.trims = min(self.trims, (self.inferencesPerRound - 1) / 2)

        runAndShowBenchmark(modelSel: self.modelSelection)
    }

    @IBAction func settingButtonTouched(_ sender: Any) {
        print("settingButtonTouched")
        settingMenuView.isHidden = !settingMenuView.isHidden
    }

    @IBAction func menuItemValueChanged(_ sender: UITextField!)
    {
        if let value = Int(sender.text ?? "0") {
            if sender.tag == 0 {
                self.rounds = value
            } else if sender.tag == 1 {
                self.inferencesPerRound = value
            } else if sender.tag == 2 {
                self.trims = value
            } else {
                print("UISwitch tag not known")
            }
        }
    }

    // Main function to perform a benchmark session
    func runAndShowBenchmark(modelSel: ModelType) {
        inferenceQueue.async { [self] in
            self.latencies.reserveCapacity(self.inferencesPerRound)

            let attributes = CFDictionaryCreateMutable(kCFAllocatorDefault, 1, nil, nil);
            var pixelBuffer224x224: CVPixelBuffer? = nil
            let _: CVReturn = CVPixelBufferCreate(kCFAllocatorDefault, 224, 224,
                kCVPixelFormatType_32BGRA,
                attributes, &pixelBuffer224x224)

            var mobileOneS0_input = mobileone_s0Input(input: pixelBuffer224x224!)
            var mobileOneS1_input = mobileone_s1Input(input: pixelBuffer224x224!)
            var mobileOneS2_input = mobileone_s2Input(input: pixelBuffer224x224!)
            var mobileOneS3_input = mobileone_s3Input(input: pixelBuffer224x224!)
            var mobileOneS4_input = mobileone_s4Input(input: pixelBuffer224x224!)

            for _ in 0..<self.rounds {

                self.latencies.removeAll()

                switch (modelSel) {
                case .mobileOneS0:
                    modelMobileOneS0 = try! mobileone_s0(configuration: MLModelConfiguration())
                    mobileOneS0_input = mobileone_s0Input(input: pixelBuffer224x224!)
                case .mobileOneS1:
                    modelMobileOneS1 = try! mobileone_s1(configuration: MLModelConfiguration())
                    mobileOneS1_input = mobileone_s1Input(input: pixelBuffer224x224!)
                case .mobileOneS2:
                    modelMobileOneS2 = try! mobileone_s2(configuration: MLModelConfiguration())
                    mobileOneS2_input = mobileone_s2Input(input: pixelBuffer224x224!)
                case .mobileOneS3:
                    modelMobileOneS3 = try! mobileone_s3(configuration: MLModelConfiguration())
                    mobileOneS3_input = mobileone_s3Input(input: pixelBuffer224x224!)
                case .mobileOneS4:
                    modelMobileOneS4 = try! mobileone_s4(configuration: MLModelConfiguration())
                    mobileOneS4_input = mobileone_s4Input(input: pixelBuffer224x224!)
                }

                for inference in -self.warmUps..<self.inferencesPerRound {

                    let predictStart = Date()

                    switch (modelSel) {
                    case .mobileOneS0:
                        let _ = try! modelMobileOneS0.prediction(input: mobileOneS0_input)
                    case .mobileOneS1:
                        let _ = try! modelMobileOneS1.prediction(input: mobileOneS1_input)
                    case .mobileOneS2:
                        let _ = try! modelMobileOneS2.prediction(input: mobileOneS2_input)
                    case .mobileOneS3:
                        let _ = try! modelMobileOneS3.prediction(input: mobileOneS3_input)
                    case .mobileOneS4:
                        let _ = try! modelMobileOneS4.prediction(input: mobileOneS4_input)
                    }

                    let predictTime = 1000.0 * Date().timeIntervalSince(predictStart)

                    // ignore the warmup benchmark inferences result
                    if inference < 0 {
                        continue
                    }
                    self.latencies.append(predictTime)
                }

                self.updateLatencies()
                self.updateLatencyStatsDisplay()
            }

            self.updateLatencyValuesDisplay()
        }
    }

    func updateLatencies() {

        // Sort then trim latencies
        self.latencies.sort()
        let trimmed_latencies = latencies[self.trims..<(self.inferencesPerRound - self.trims)]

        // Calculate latency averages.
        self.averageLatency =
            trimmed_latencies.reduce(0.0, +) / Double(trimmed_latencies.count)
        self.averagies.append(self.averageLatency)
        self.averageAll = self.averagies.reduce(0.0, +) / Double(self.averagies.count)

        if let low = trimmed_latencies.first {
            self.latencyLow = min(self.latencyLow, low)
        }
        if let high = trimmed_latencies.last {
            self.latencyHigh = max(self.latencyHigh, high)
        }

        self.roundsLatencies.append(self.latencies)
    }

    func updateLatencyStatsDisplay() {
        DispatchQueue.main.async {
            self.averageAllLatencyField.text = String(format: "%.3f", self.averageAll)
            self.averageLastLatencyField.text = String(format: "%.3f", self.averageLatency)
            self.latencyLowField.text = String(format: "%.3f", self.latencyLow)
            self.latencyHighField.text = String(format: "%.3f", self.latencyHigh)
        }
    }

    func updateLatencyValuesDisplay() {

        let trimmedColorAttribute = [NSAttributedString.Key.foregroundColor: UIColor.gray]
        let inMeanColorAttribute = [NSAttributedString.Key.foregroundColor: UIColor.green]
        let endColorAttribute = [NSAttributedString.Key.foregroundColor: UIColor.white]
        let nsSep = NSMutableAttributedString(string: ", ", attributes: trimmedColorAttribute)
        let nsBegin = NSMutableAttributedString(string: "[ ", attributes: endColorAttribute)
        let nsEnd = NSMutableAttributedString(string: " ], \n", attributes: endColorAttribute)

        DispatchQueue.main.async {
            let valuesAttributesText = NSMutableAttributedString()

            // Display benchmark values in each round in colored attributes text within squre braskets.
            for round in 0..<self.roundsLatencies.count {
                let latenciesStrings = self.roundsLatencies[round].map { String(format: "%.3f", $0) }

                let allAttrStr = NSMutableAttributedString()
                allAttrStr.append(nsBegin)
                for n in 0..<latenciesStrings.count {
                    let attrLatencyStr: NSMutableAttributedString
                    if n < self.trims || n >= self.inferencesPerRound - self.trims {
                        attrLatencyStr =
                            NSMutableAttributedString(string: latenciesStrings[n],
                            attributes: trimmedColorAttribute)
                    } else {
                        attrLatencyStr =
                            NSMutableAttributedString(string: latenciesStrings[n],
                            attributes: inMeanColorAttribute)
                    }
                    allAttrStr.append(attrLatencyStr)
                    if n < self.inferencesPerRound - 1 {
                        allAttrStr.append(nsSep)
                    } else {
                        allAttrStr.append(nsEnd)
                    }
                }

                valuesAttributesText.append(allAttrStr)
            }
            self.latenciesTextView.attributedText = valuesAttributesText

            self.runButton.isEnabled = true
        }
    }

    func setupSettingMenuView() {
        settingMenuViewDataSource = [
            ("Rounds", self.rounds, 0),
            ("Inferences per Round", self.inferencesPerRound, 1),
            ("Low/High Trim", self.trims, 2)
        ]
        settingMenuView.translatesAutoresizingMaskIntoConstraints = false
        settingMenuView.backgroundColor = UIColor.systemGray.withAlphaComponent(0.90)
        settingMenuView.layer.borderWidth = 1
        settingMenuView.layer.borderColor = UIColor.yellow.cgColor
        settingMenuView.layer.cornerRadius = 5
        settingMenuView.allowsSelection = false

        self.view.addSubview(settingMenuView)
        settingMenuView.frame = CGRect(x: 80, y: 100, width: 200, height: 100)

        settingMenuView.isScrollEnabled = true
        settingMenuView.delegate = self
        settingMenuView.dataSource = self
        settingMenuView.register(SettingMenuViewCell.self,
            forCellReuseIdentifier: "SettingMenuViewCell")

        let widthConstraint = NSLayoutConstraint(
            item: settingMenuView, attribute: .width, relatedBy: .greaterThanOrEqual, toItem: nil,
            attribute: .notAnAttribute, multiplier: 1, constant: 220)
        let heightConstraint = NSLayoutConstraint(
            item: settingMenuView, attribute: .height, relatedBy: .greaterThanOrEqual, toItem: nil,
            attribute: .notAnAttribute, multiplier: 1,
            constant: CGFloat(40 * settingMenuViewDataSource.count))
        let leadingContraint = NSLayoutConstraint(
            item: settingMenuView, attribute: .leading, relatedBy: .equal, toItem: settingButton,
            attribute: .leading, multiplier: 1, constant: 0)
        let bottomContraint = NSLayoutConstraint(
            item: settingMenuView, attribute: .bottom, relatedBy: .equal, toItem: settingButton,
            attribute: .top, multiplier: 1, constant: 0)

        view.addConstraints([widthConstraint, heightConstraint, leadingContraint, bottomContraint])
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        self.view.endEditing(true)
    }

    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        settingMenuViewDataSource.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        guard let cell = tableView
            .dequeueReusableCell(withIdentifier: "SettingMenuViewCell",
            for: indexPath) as? SettingMenuViewCell
            else {
            fatalError("unable to deque SettingMenuViewCell")
        }

        let dataSource = settingMenuViewDataSource[indexPath.row]
        cell.labelView.text = dataSource.0
        cell.valueView.text = String(dataSource.1)
        cell.valueView.tag = dataSource.2
        cell.valueView.addTarget(self, action: #selector(menuItemValueChanged(_:)), for: UIControl.Event.editingChanged)
        cell.valueView.delegate = self
        cell.valueView.returnKeyType = .done
        cell.backgroundView = nil
        cell.backgroundColor = .clear
        cell.selectionStyle = .blue

        return cell
    }

    func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
        return 40
    }

    func textFieldShouldReturn(_ textField: UITextField) -> Bool
    {
        textField.resignFirstResponder()
        return true
    }

    func menuModelHandler(action: UIAction) {
        if let modelType = ModelType(rawValue: action.title) {
            self.modelSelection = modelType
        } else {
            print("Model Menu Action is wrong: '\(action.title)'")
        }
    }

}

