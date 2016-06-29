//
//  ViewController.swift
//  IntegralImage
//
//  Created by Christopher Helf on 28.06.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let tests = TestClass()
        assert(tests.testSmallTextureSum() == true)
        assert(tests.compareImplAgainstMPSWithBounds() == true)
        assert(tests.compareImplAgainstMPS() == true)
        assert(tests.testMPS() == true)
        assert(tests.testTimes720p() == true)
        assert(tests.testTimes1080p() == true)
        
        
        //printTexture(texture: output)
        
        print("Tests Completed")
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

