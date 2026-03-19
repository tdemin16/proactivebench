import copy
import inspect
import os
import random
import re

STOP = 0
LEFT = 1
RIGHT = 2
REWIND = 3
FORWARD = 4
ROTATE = 5
BRIGHTNESS = 6
CONTRAST = 7
SATURATE = 8
BLUR = 9
NOISE = 10
JPEG_COMPRESSION = 11
PIXELATE = 12
ARTIFACTS = 13
DETAILS = 14
UP = 15
DOWN = 16
ZOOM = 17


def get_environment(dataset: str):
    match dataset:
        case "Realistic-Occlusion-Dataset":
            return RealisticOcclusionDatasetEnvironment
        case "OcclusionDataSet-MM20":
            return OcclusionDataSetMM20Environment
        case "MVP-N":
            return MVPNEnvironment
        case "ImageNet-C":
            return ImageNetCEnvironment
        case "QuickDraw":
            return QuickDrawEnvironment
        case "ChangeIt":
            return ChangeItEnvironment
        case "coco2014":
            return COCO2014Environment
        case _:
            raise NotImplementedError(dataset)


class BaseEnvironment:
    def __init__(
        self,
        entry: dict,
        data_dir: str,
        resize_samples: bool = False,
        reference: bool = False,
        random: bool = False,
        verbose: bool = False,
        _valid_images: list = [],
        _pre_prompt: str = "",
        _reference_prompt: str = None,
        _hint_prompt: str = None,
        _cot_prompt: str = None,
        _index_correct_options: list = None,
        _action_space: tuple = None,
        _max_steps: int = None,
    ):
        self.id = entry["id"]

        # images info
        self.image_template = entry["image_template"]  # relative image path template
        self.first_image = entry["first_image"]  # index of first image (7)
        self.num_images = entry["num_images"]  # number of images in the dataset
        self.valid_images = _valid_images if _valid_images != [] else list(range(self.num_images))
        self.data_dir = data_dir  # dataset root directory
        self.resize_samples = resize_samples  # whether to resize samples to avoid oom issues (used some models with history)

        # reference (no occlusions) input info
        # objective: assess model accuracy with no occlusions
        self.reference_image = entry["reference_image"]
        self.reference_prompt = _reference_prompt
        self.reference = reference  # when true only return the reference image and stop evaluation

        # prompts and vqa options
        self.pre_prompt = _pre_prompt  # static prompt part (avoid storing redundant prompts)
        self.prompt = entry["prompt"]  # user prompt
        self.hint_prompt = _hint_prompt
        self.cot_prompt = _cot_prompt
        self.options = entry["options"]  # interactions options
        self.categories = entry["categories"]  # possible categories
        self.answer = entry["answer"]  # correct category
        self.index_correct_options = _index_correct_options
        self.correct_options = [
            opt for i, opt in enumerate(self.options) if i in self.index_correct_options
        ]
        self.random = random

        # index of the correct category
        self.index_correct_category = self.categories.index(self.answer)
        self.correct_categories = [self.answer]  # create this list just for generalize evaluate

        # environment state
        self.action_space = (
            _action_space  # possible actions space, predicting the correct answer lead to a stop
        )
        self.action_map = {
            "to the left": LEFT,
            "to the right": RIGHT,
            "rewind the video.": REWIND,
            "wait for the": FORWARD,
            "rotate": ROTATE,
            "view": ROTATE,
            "angle": ROTATE,
            "orientation": ROTATE,
            "different": ROTATE,
            "brightness": BRIGHTNESS,
            "contrast": CONTRAST,
            "saturation": SATURATE,
            "deblur": BLUR,
            "denoise": NOISE,
            "compression": JPEG_COMPRESSION,
            "resolution": PIXELATE,
            "artifacts": ARTIFACTS,
            "details": DETAILS,
            "quickdraw": DETAILS,
            "drawing": DETAILS,
            "sketch": DETAILS,
            "up": UP,
            "down": DOWN,
            "farther": ZOOM,
        }
        self.action = None  # stores action computed by LLM
        self.image_id = self.first_image  # image_id of next image
        self.stop = False  # wheter to stop
        self.max_steps = _max_steps
        assert self._check_substrings()  # check for errors in action_map

        # statistics
        self.num_turns = 0
        self.generated_answer_history = []
        self.action_evaluation_history = []
        self.category_evaluation_history = []
        self.action_history = []
        self.image_id_history = []
        self.correct_prediction = False

        # other
        self.verbose = verbose

        if self.verbose:
            print(self)

    def get_state(self, hint: bool = False):
        """
        Returns:
        image_path: str
            Full path of current image
        prompt: str
            User prompt
        options: str
            Available answers to :prompt:
        """

        # case reference image (no occlusions)
        if self.reference:
            # build image path
            relative_image_path = self._get_relative_image_path(image_id=self.reference_image)
            image_path = os.path.join(self.data_dir, relative_image_path)

            # get prompts and categories
            prompt = self.reference_prompt if self.reference_prompt is not None else self.prompt
            options = self.categories

        # case occlusion
        else:
            # build image path
            relative_image_path = self._get_relative_image_path(image_id=self.image_id)
            image_path = os.path.join(self.data_dir, relative_image_path)

            # one image is corrupted, return a different one
            image_path = self._fix_corruptions(image_path)

            # get prompt and build options using self.options and self.categories
            # we get the valid options as these can change through time
            prompt = self.prompt
            options = self._get_valid_options() + self.categories

        # add pre-prompt and hint (if true)
        if self.pre_prompt != "":
            prompt = f"{self.pre_prompt} {prompt}"
        if hint:
            prompt = f"{prompt} {self.hint_prompt}"

        if self.verbose:
            print(
                f"[LOG] get_state(hint={hint}) -> image_path: {image_path} - prompt: {prompt} - options: {options}"
            )

        transform = self._get_transform()
        if self.verbose:
            print(f"[LOG] _get_transform() -> \n{inspect.getsource(transform)}")

        return {
            "image_path": image_path,
            "prompt": prompt,
            "options": options,
            "transform": transform,
        }

    def _get_valid_options(self):
        return self.options

    def get_icl_example(
        self,
        hint: bool = False,
        cot: bool = False,
        resize: bool = False,
        reduce_icl_steps: bool = False,
    ):
        raise NotImplementedError

    def _get_transform(self):
        if self.resize_samples:

            def transform(image):
                short_edge = min(image.size)
                scale = 224 / short_edge
                new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                return image.resize(new_size)

        else:

            def transform(image):
                return image

        return transform

    def _evaluate(
        self, generated_answer: str, correct_answers: list, options: list, shift_index: int
    ):
        """
        returns "correct" if correct answer, "wrong" if wrong answer but in options, "mismatch" if answer is not within options.
        returns answer. Always use the "options" list as it contains the correct index
        """

        # try with full answer
        full_answer = self._standardize_full_answer(generated_answer)
        std_correct_answer = [self._standardize_full_answer(ans) for ans in correct_answers]
        std_options = [self._standardize_full_answer(option) for option in options]

        if self.verbose:
            print(
                f"[LOG] _evaluate({generated_answer}, {correct_answers}, {options}, {shift_index})"
            )
            print("[LOG] try full answer")
            print("[LOG] full_answer", full_answer)
            print("[LOG] std_correct_answer", std_correct_answer)
            print("[LOG] std_options", std_options)

        # if correct answer then "cool!"
        if full_answer in std_correct_answer:
            index = std_options.index(full_answer)
            if self.verbose:
                print("[LOG] full_answer is correct")
            return "correct", options[index]

        # if answer in option, we don't need to check for letters, let's return "wrong"
        if full_answer in std_options:
            index = std_options.index(full_answer)
            if self.verbose:
                print("[LOG] full_answer is wrong")
            return "wrong", options[index]

        # we can't return "mismatch" yet as we should check letters first
        # thus, let's try with letters
        letter_answer = self._standardize_letter_answer(generated_answer)
        std_correct_letters = [
            self._standardize_letter_answer(chr(65 + options.index(ans) + shift_index))
            for ans in correct_answers
        ]
        std_options_letters = [
            self._standardize_letter_answer(chr(65 + i + shift_index)) for i in range(len(options))
        ]

        if self.verbose:
            print("[LOG] try letter")
            print("[LOG] letter_answer", letter_answer)
            print("[LOG] std_correct_letters", std_correct_letters)
            print("[LOG] std_options_letters", std_options_letters)

        # if correct letter then "cool!"
        if letter_answer in std_correct_letters:
            index = std_options_letters.index(letter_answer)
            if self.verbose:
                print("[LOG] letter_answer is correct")
            return "correct", options[index]

        # if answer in correct range, let's return "wrong"
        if letter_answer in std_options_letters:
            index = std_options_letters.index(letter_answer)
            if self.verbose:
                print("[LOG] letter_answer is wrong")
            return "wrong", options[index]

        if self.verbose:
            print("[LOG] mismatch")
        return "mismatch", ""

    def _standardize_full_answer(self, answer):
        """
        - remove leading chars followed by a "." (e.g., A. answer)
        - lowers the string avoiding case sensitive issues
        - remove leading and trailing white spaces
        - remove all white spaces (there could be additional ones)
        """
        answer = re.sub(r"^.*[A-Z]\.\s*", "", answer)
        return answer.lower().strip().replace(" ", "")

    def _standardize_letter_answer(sefl, answer):
        """
        If valid letter answer return the letter, otherwise return "#"
        - split over "." and take index 0
        - lowers the string avoiding case sensitive issues
        - remove leading and trailing white spaces
        """
        answer = answer.split(".")[0].lower().strip()
        if len(answer) > 1:
            return "#"
        return answer

    def evaluate_generated_answer(self, generated_answer: str):
        correct_options, correct_categories = self._get_correct_options_and_categories()
        if self.reference:
            action_evaluation = "mismatch"
            category_evaluation, chosen_category = self._evaluate(
                generated_answer, correct_categories, self.categories, shift_index=0
            )
        else:
            action_evaluation, chosen_action = self._evaluate(
                generated_answer,
                correct_options,
                self._get_valid_options(),
                shift_index=0,
            )
            category_evaluation, chosen_category = self._evaluate(
                generated_answer,
                correct_categories,
                self.categories,
                # if there are options, category letters will no correspond to their index and must be shifted
                shift_index=len(self._get_valid_options()),
            )

        # if both answers are either wrong or not in the considered answers
        # set the next action to STOP
        if action_evaluation in ("wrong", "mismatch") and category_evaluation in (
            "wrong",
            "mismatch",
        ):
            self.action = STOP

        # if correct category then stop and set correct_prediction to True
        elif category_evaluation == "correct":
            self.correct_prediction = True
            self.action = STOP
            assert (
                chosen_category == self.answer
            ), f"chosen_category {chosen_category} - answer {self.answer}"

        # if correct action then map it to action index
        elif action_evaluation == "correct":
            self._parse_action(chosen_action)

        else:
            raise ValueError(f"Something is wrong. Action: {self.action}.")

        # update internal statistics
        self._update_statistics(
            generated_answer=generated_answer,
            action_evaluation=action_evaluation,
            category_evaluation=category_evaluation,
        )

        # check max_step condition
        self._check_max_steps()

        # if we should continue, let's prepare the next state
        if self.action != STOP:
            self._prepare_next_state()
            assert (
                self.image_id in self.valid_images
            ), f"valid images: {self.valid_images} - image id: {self.image_id}"

        # set the stop condition
        self._set_stop()

        if self.verbose:
            print(
                f"[LOG] evaluate_generated_answer({generated_answer}) -> correct_options: {correct_options} - options: {self.options} - correct_categories: {self.correct_categories} - categories: {self.categories} - action_evaluation: {action_evaluation} - category_evaluation: {category_evaluation} - action: {self.action} - num_turns: {self.num_turns} - image_id: {self.image_id} - stop: {self.stop} - correct_prediction: {self.correct_prediction}"
            )

    def get_statistics(self):
        return {
            "num_turns": self.num_turns,
            "generated_answers": self.generated_answer_history,
            "action_evaluations": self.action_evaluation_history,
            "category_evaluations": self.category_evaluation_history,
            "correct_prediction": self.correct_prediction,
            "image_id_history": self.image_id_history,
            "action_history": self.action_history,
            "max_steps": self.num_turns == self.max_steps,
        }

    def get_open_ended_gen_answers(self):
        raise NotImplementedError

    def _get_relative_image_path(self, image_id):
        raise NotImplementedError

    def _fix_corruptions(self, image_path: str):
        return image_path

    def _get_correct_options_and_categories(self):
        """
        Returns the list of correct options and categories for the give state
        """
        raise NotImplementedError

    def _parse_action(self, chosen_action: str):
        chosen_action = chosen_action.lower()

        # if random, chose a valid answer to continue evaluation
        if self.random:
            admissible_actions = list(set(self.action_space).difference({STOP}))
            chosen_action = random.choice(
                [
                    possible_answer
                    for possible_answer, action_number in self.action_map.items()
                    if action_number in admissible_actions
                ]
            )

        # check which action does the chosen answer belongs to
        found_match = False
        for key, value in self.action_map.items():
            # limit search to action_space
            if value in self.action_space:
                # let's be sure that there are no duplicate keys
                # this assert works as follows:
                # 1. ok if the chosen action does not imply the investigated action (key)
                # 2. ok if the chosen action implies the investigated action, but no action was set (not found match)
                # 3. ok if the chosen action implies the investigated action, the action was already set, but the investigated adction and the one previously set are the same -> rotation
                assert (
                    key not in chosen_action
                    or not found_match
                    or self.action == value
                    and self.action == ROTATE
                ), f"chosen_action {chosen_action} - key {key} - key in chosen_action {key in chosen_action} - found_match {found_match} - action {self.action}."
                if key in chosen_action and not found_match:
                    self.action = value
                    found_match = True

        # the chosen_action must be associated to an action in the action_space
        assert found_match, f"chosen_action {chosen_action}, action_space {self.action_space}"
        # also check that predicted answer are in the answers space (should already be caught by above if value in self.action_space)
        assert (
            self.action in self.action_space
        ), f"chosen_answer {chosen_action} - action {self.action} - action_space {self.action_space}."

    def _update_statistics(self, generated_answer, action_evaluation, category_evaluation):
        self.num_turns += 1
        self.generated_answer_history.append(generated_answer)
        self.action_evaluation_history.append(action_evaluation)
        self.category_evaluation_history.append(category_evaluation)
        self.image_id_history.append(self.image_id)
        self.action_history.append(self.action)

    def _check_max_steps(self):
        if self.num_turns == self.max_steps:
            self.action = STOP

    def _prepare_next_state(self):
        raise NotImplementedError

    def _set_stop(self):
        if self.action == STOP:
            self.stop = True

    def _check_substrings(self):
        """
        Check if the i-th string is contained in the j-th.
        Need to eval entire matrix except for diagonal
        """
        keys = list(self.action_map.keys())
        for i in range(len(keys)):
            for j in range(len(keys)):
                if i != j and keys[i] in keys[j]:
                    print(f"[bold red][Error][/]key {keys[i]} in key {keys[j]}")
                    return False
        return True

    def __repr__(self):
        repr = f"{type(self).__name__}(\n"
        for attr, value in self.__dict__.items():
            repr += f"    {attr}={value}\n"
        repr += ")"
        return repr


class RealisticOcclusionDatasetEnvironment(BaseEnvironment):
    def __init__(
        self,
        entry: dict,
        data_dir: str,
        resize_samples: bool = False,
        reference: bool = False,
        random: bool = False,
        verbose: bool = False,
    ):
        reference_prompt = "What type of object is in the image?"  # from ale's paper
        hint_prompt = "Hint: moving the occluding object might reveal what is behind it."
        cot_prompt = "I cannot see what is behind the occluding object. Moving the occluding object might reveal its position. So, first, I need to move the occluding object and then answer the question. Therefore, the answer is:"
        index_correct_options = set(  # indexes of options that are admitted
            [
                i
                for i, opt in enumerate(entry["options"])
                if "cannot" not in opt.lower() and "do not" not in opt.lower()
            ]
        )
        action_space = (STOP, LEFT, RIGHT)
        max_steps = entry["num_images"]

        super().__init__(
            entry=entry,
            data_dir=data_dir,
            resize_samples=resize_samples,
            reference=reference,
            random=random,
            verbose=verbose,
            _reference_prompt=reference_prompt,
            _hint_prompt=hint_prompt,
            _cot_prompt=cot_prompt,
            _index_correct_options=index_correct_options,
            _action_space=action_space,
            _max_steps=max_steps,
        )

    def _get_resize_transform(self):
        def transform(image):
            short_edge = min(image.size)
            scale = 512 / short_edge
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            return image.resize(new_size)

        return transform

    def get_icl_example(
        self,
        hint: bool = False,
        cot: bool = False,
        resize: bool = False,
        reduce_icl_steps: bool = False,
    ):
        assert (
            not self.reference
        ), "Although it makes sense, it's not implemented for reference == True :c"

        if self.verbose:
            print(f"[LOG] get_icl_example(resize={resize})")

        # get step 0
        step_0 = self.get_state(hint)
        if resize:
            step_0["transform"] = self._get_resize_transform()
            if self.verbose:
                print(
                    f"[LOG] _get_resize_transform() -> \n{inspect.getsource(step_0["transform"])}"
                )

        # sample correct option
        option_index = random.choice(list(self.index_correct_options))
        chosen_option = self.options[option_index]

        # add cot prompting
        if cot:
            chosen_option = f"{self.cot_prompt} {chr(65+option_index)}. {chosen_option}"

        step_0["generated_answer"] = chosen_option

        # act on environment
        self.evaluate_generated_answer(chosen_option)

        # get step 1
        step_1 = self.get_state(hint)
        if resize:
            step_1["transform"] = self._get_resize_transform()
            if self.verbose:
                print(
                    f"[LOG] _get_resize_transform() -> \n{inspect.getsource(step_1["transform"])}"
                )
        step_1["generated_answer"] = self.answer

        if self.verbose:
            print(f"[LOG] -> step_0: {step_0} - step_1: {step_1}")

        return [step_0, step_1]

    def get_open_ended_gen_answers(self):
        # if I cannot see what's behind something, I just change my pov or remove the occlusion
        return "Move the blocks, Move the camera."

    def _get_relative_image_path(self, image_id):
        return self.image_template.format(image=image_id)

    def _fix_corruptions(self, image_path: str):
        if "natural_bluebl_greenr-banana-s6" in image_path:
            if self.action == LEFT:
                return image_path.replace("s6", "s5")
            else:
                return image_path.replace("s6", "s7")
        return image_path

    def _get_correct_options_and_categories(self):
        options = []
        categories = []
        if self.reference:
            categories = self.correct_categories

        # step 0
        elif self.action == None:
            # if step 0, correct options are only those in self.options (it is unfeasible to answer otherwise)
            # here we compute the list of admissible options
            options = self.correct_options

        # step > 0
        else:
            options = self.correct_options
            categories = self.correct_categories

        return options, categories

    def _prepare_next_state(self):
        image_id = self.image_id - 1 if self.action == LEFT else self.image_id + 1
        self.image_id = image_id % self.num_images


class OcclusionDataSetMM20Environment(BaseEnvironment):
    def __init__(
        self,
        entry: dict,
        data_dir: str,
        resize_samples: bool = False,
        reference: bool = False,
        random: bool = False,
        verbose: bool = False,
    ):
        # ? for this dataset we need a long list of steps, it's not easy and cot does not really work for small models
        pre_prompt = "This is a frame extracted from a video. Answer the following question."
        hint_prompt = "Hint: If there is an occlusion, waiting for it to disappear or rewinding the video might reveal what's behind it."
        index_correct_options = set(
            [
                i
                for i, opt in enumerate(entry["options"])
                if "cannot" not in opt.lower() and "do not" not in opt.lower()
            ]
        )
        action_space = (STOP, REWIND, FORWARD)
        max_steps = entry["num_images"]

        self.strong_occlusions = entry["strong_occlusions"]
        super().__init__(
            entry=entry,
            data_dir=data_dir,
            resize_samples=resize_samples,
            reference=reference,
            random=random,
            verbose=verbose,
            _pre_prompt=pre_prompt,
            _hint_prompt=hint_prompt,
            _index_correct_options=index_correct_options,
            _action_space=action_space,
            _max_steps=max_steps,
        )

    def get_open_ended_gen_answers(self):
        # if someone is in front of a camera I can move the camera or wait for him to move
        # as it's a video, we can alswo rewind it.
        # also there are cases where the model can already answer
        options, categories = self._get_correct_options_and_categories()
        categories = ", ".join([cat.replace(".", "") for cat in categories])
        answers = (
            "Wait for the occlusion to disappear, Rewind the video, Move the camera, "
            + categories
            + "."
        )
        return answers

    def _get_relative_image_path(self, image_id):
        return self.image_template.format(image=str(image_id).zfill(4))

    def _fix_corruptions(self, image_path):
        # ? Until no corruption found, let's do nothing
        return super()._fix_corruptions(image_path)

    def _get_correct_options_and_categories(self):
        options = []
        categories = []
        if self.reference:
            categories = self.correct_categories

        else:
            options = self.correct_options
            categories = self.correct_categories

        return options, categories

    def _prepare_next_state(self):
        image_id = self.image_id - 1 if self.action == REWIND else self.image_id + 1
        self.image_id = image_id % self.num_images


class MVPNEnvironment(BaseEnvironment):
    def __init__(
        self,
        entry: dict,
        data_dir: str,
        resize_samples: bool = False,
        reference: bool = False,
        random: bool = False,
        verbose: bool = False,
    ):
        valid_images = entry["valid_images"]
        hint_prompt = "Hint: rotating the object could provide a more informative view."
        cot_prompt = "I cannot tell which object is this. Rotating the object or changing the viewpoint might provide a more informative view. So, first, I need to rotate the object or change the viewpoint and then answer the question. Therefore, the answer is:"
        index_correct_options = set(
            [
                i
                for i, opt in enumerate(entry["options"])
                if "cannot" not in opt.lower() and "do not" not in opt.lower()
            ]
        )
        action_space = (STOP, ROTATE)
        max_steps = entry["num_images"]

        super().__init__(
            entry=entry,
            data_dir=data_dir,
            resize_samples=resize_samples,
            reference=reference,
            random=random,
            verbose=verbose,
            _valid_images=valid_images,
            _hint_prompt=hint_prompt,
            _cot_prompt=cot_prompt,
            _index_correct_options=index_correct_options,
            _action_space=action_space,
            _max_steps=max_steps,
        )

    def _get_resize_transform(self):
        def transform(image):
            short_edge = min(image.size)
            scale = 112 / short_edge
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            return image.resize(new_size)

        return transform

    def get_icl_example(
        self, hint=False, cot=False, resize: bool = False, reduce_icl_steps: bool = False
    ):
        assert (
            not self.reference
        ), "Although it makes sense, it's not implemented for reference == True :c"

        if self.verbose:
            print(f"[LOG] get_icl_example(hint=hint)")

        steps = []
        while not self.stop:
            # get i-th step
            step = self.get_state(hint)
            if resize:
                step["transform"] = self._get_resize_transform()
                if self.verbose:
                    print(
                        f"[LOG] _get_resize_transform() -> \n{inspect.getsource(step["transform"])}"
                    )

            # get correct option
            if self.num_turns + 1 < self.num_images:
                # all images before last are non informative, we must rotate
                option_index = list(self.index_correct_options)[0]
                chosen_option = self.options[option_index]

                # add cot prompting
                if cot:
                    chosen_option = f"{self.cot_prompt} {chr(65+option_index)}. {chosen_option}"

            else:
                chosen_option = self.answer

            step["generated_answer"] = chosen_option

            # act on environment
            self.evaluate_generated_answer(chosen_option)

            # record step
            steps.append(step)

        len_steps = len(steps)
        if reduce_icl_steps and len_steps > 2:
            steps = [steps[0], steps[-1]]
            if self.verbose:
                print(f"[LOG] reduce_icl_steps: {len_steps} -> {len(steps)}")

        return steps

    def get_open_ended_gen_answers(self):
        # either I rotate the object or I can think of moving the camera
        # the view is uninformative, thus it cannot answer
        return "Rotate the object, Move the camera."

    def _get_relative_image_path(self, image_id):
        informative = int(self.num_turns + 1 == self.num_images) or int(self.reference)
        return self.image_template.format(informative=informative, image=str(image_id).zfill(4))

    def _fix_corruptions(self, image_path):
        # ? Until no corruption found, let's do nothing
        return super()._fix_corruptions(image_path)

    def _get_correct_options_and_categories(self):
        options = []
        categories = []
        if self.reference:
            categories = self.correct_categories

        # llm is allowed to rotate on all steps before the last one
        elif self.num_turns + 1 < self.num_images:
            options = self.correct_options
            categories = self.correct_categories

        # if last step, only correct option is to predict category
        elif self.num_turns + 1 == self.num_images:
            categories = self.correct_categories

        else:
            raise ValueError(
                f"reference {self.reference} - num_turns {self.num_turns} - num_images {self.num_images}"
            )

        return options, categories

    def _prepare_next_state(self):
        self.image_id = self.valid_images[self.num_turns]


class ImageNetCEnvironment(BaseEnvironment):
    def __init__(
        self,
        entry: dict,
        data_dir: str,
        resize_samples: bool = False,
        reference: bool = False,
        random: bool = False,
        verbose: bool = False,
    ):
        valid_images = list(range(1, 6))
        # reference prompt same as normal one
        hint_prompt = "Hint: enhancing the image quality could help with classification."
        # ? cot and icl hard for this dataset
        try:
            index_correct_options = set([entry["options"].index(entry["quality_answer"])])
        except ValueError:
            # handle random case
            assert random
            index_correct_options = set(
                [
                    i
                    for i, opt in enumerate(entry["options"])
                    if "cannot" not in opt.lower() and "do not" not in opt.lower()
                ]
            )

        action_space = (
            STOP,
            BRIGHTNESS,
            CONTRAST,
            SATURATE,
            BLUR,
            NOISE,
            JPEG_COMPRESSION,
            PIXELATE,
            ARTIFACTS,
        )
        max_steps = entry["num_images"]

        super().__init__(
            entry=entry,
            data_dir=data_dir,
            resize_samples=resize_samples,
            reference=reference,
            random=random,
            verbose=verbose,
            _valid_images=valid_images,
            _hint_prompt=hint_prompt,
            _index_correct_options=index_correct_options,
            _action_space=action_space,
            _max_steps=max_steps,
        )

    def get_open_ended_gen_answers(self):
        # here it should either predict the correct distortion or the correct class
        options, categories = self._get_correct_options_and_categories()
        answers = ", ".join([a.replace(".", "") for a in options + categories]) + "."
        return answers

    def _get_relative_image_path(self, image_id):
        return self.image_template.format(level=image_id)

    def _fix_corruptions(self, image_path):
        # ? until no corruptions everything is fine
        return super()._fix_corruptions(image_path)

    def _get_correct_options_and_categories(self):
        options = []
        categories = []
        if self.reference:
            categories = self.correct_categories

        # llm is allowed to improve image quality on all steps before the last one
        elif self.num_turns + 1 < self.num_images:
            options = self.correct_options
            categories = self.correct_categories

        # if last step, only correct option is to predict category
        elif self.num_turns + 1 == self.num_images:
            categories = self.correct_categories

        else:
            raise ValueError(
                f"reference {self.reference} - num_turns {self.num_turns} - num_images {self.num_images}"
            )

        return options, categories

    def _prepare_next_state(self):
        image_id = self.image_id - 1
        self.image_id = image_id


class QuickDrawEnvironment(BaseEnvironment):
    def __init__(
        self,
        entry: dict,
        data_dir: str,
        resize_samples: bool = False,
        reference: bool = False,
        random: bool = False,
        verbose: bool = False,
    ):
        hint_prompt = "Hint: Adding more details to the quickdraw could help with classification."
        index_correct_options = set(
            [
                i
                for i, opt in enumerate(entry["options"])
                if "cannot" not in opt.lower() and "do not" not in opt.lower()
            ]
        )
        action_space = (STOP, DETAILS)
        max_steps = entry["num_images"]

        super().__init__(
            entry=entry,
            data_dir=data_dir,
            resize_samples=resize_samples,
            reference=reference,
            random=random,
            verbose=verbose,
            _hint_prompt=hint_prompt,
            _index_correct_options=index_correct_options,
            _action_space=action_space,
            _max_steps=max_steps,
        )

    def get_open_ended_gen_answers(self):
        # llm can ask for improvements in the drawing or predict category
        options, categories = self._get_correct_options_and_categories()
        categories = ", ".join([cat.replace(".", "") for cat in categories])
        return "Add more details to the drawing, " + categories + "."

    def _get_relative_image_path(self, image_id):
        return self.image_template.format(image=str(image_id).zfill(4))

    def _fix_corruptions(self, image_path):
        # ? Until no corruption found, let's do nothing
        return super()._fix_corruptions(image_path)

    def _get_correct_options_and_categories(self):
        options = []
        categories = []
        if self.reference:
            categories = self.correct_categories

        # llm is allowed to add details on all steps before the last one
        elif self.num_turns + 1 < self.num_images:
            options = self.correct_options
            categories = self.correct_categories

        # if last step, only correct option is to predict category (no details left to add)
        elif self.num_turns + 1 == self.num_images:
            categories = self.correct_categories

        else:
            raise ValueError(
                f"reference {self.reference} - num_turns {self.num_turns} - num_images {self.num_images}"
            )

        return options, categories

    def _prepare_next_state(self):
        image_id = self.image_id + 1
        self.image_id = image_id


class ChangeItEnvironment(BaseEnvironment):
    def __init__(
        self,
        entry: dict,
        data_dir: str,
        resize_samples: bool = False,
        reference: bool = False,
        random: bool = False,
        verbose: bool = False,
    ):
        # ? for this dataset we need a long list of steps, it's not easy and cot does not really work for small models
        pre_prompt = "This is a frame extracted from a video. Answer the following question."
        hint_prompt = "Hint: If you cannot answer the question, waiting for it to appear or rewinding the video could help with classification."
        index_correct_options = set(
            [
                i
                for i, opt in enumerate(entry["options"])
                if "cannot" not in opt.lower() and "do not" not in opt.lower()
            ]
        )
        action_space = (STOP, REWIND, FORWARD)

        self.step_size = entry["step_size"]
        super().__init__(
            entry=entry,
            data_dir=data_dir,
            resize_samples=resize_samples,
            reference=reference,
            random=random,
            verbose=verbose,
            _pre_prompt=pre_prompt,
            _hint_prompt=hint_prompt,
            _index_correct_options=index_correct_options,
            _action_space=action_space,
            _max_steps=entry["max_steps"],
        )

    def _get_valid_options(self):
        options = copy.deepcopy(self.options)
        if not self.random:
            if self.image_id - self.step_size < 0:
                options.remove("Rewind the video.")
            elif self.image_id + self.step_size >= self.num_images:
                # either action or object
                # use for loop to find location and then remove
                for i, opt in enumerate(options):
                    if "Wait for the" in opt:
                        break
                options.remove(options[i])
        else:
            if self.verbose:
                print("[LOG] Not removing options in random")
        return options

    def get_open_ended_gen_answers(self):
        options, categories = self._get_correct_options_and_categories()
        # first frame never has rewind option
        options.append("Rewind the video.")
        answers = ",".join([a.replace(".", "") for a in options + categories]) + "."
        return answers

    def _get_relative_image_path(self, image_id):
        return self.image_template.format(image=str(image_id))

    def _fix_corruptions(self, image_path):
        # ? Until no corruption found, let's do nothing
        return super()._fix_corruptions(image_path)

    def _get_correct_options_and_categories(self):
        options = []
        categories = []
        if self.reference:
            categories = self.correct_categories

        elif self.random:
            options = self.correct_options
            categories = self.correct_categories

        # we cannot rewind the video more than this
        elif self.image_id - self.step_size < 0:
            options = copy.deepcopy(self.correct_options)
            options.remove("Rewind the video.")
            categories = self.correct_categories

        # we arrived at the end of the video, we cannot wait anymore
        elif self.image_id + self.step_size >= self.num_images:
            options = copy.deepcopy(self.correct_options)
            # either action or object
            # use for loop to find location and then remove
            for i, opt in enumerate(options):
                if "Wait for the" in opt:
                    break
            options.remove(options[i])
            categories = self.correct_categories

        else:
            options = self.correct_options
            categories = self.correct_categories

        return options, categories

    def _prepare_next_state(self):
        # we do not need to accout for the case above (first/last frame) as the action could not be selected
        # therefore we are sure that we can take the action
        image_id = (
            self.image_id - self.step_size
            if self.action == REWIND
            else self.image_id + self.step_size
        )
        # avoid unvalid states in random
        if self.random and image_id not in self.valid_images:
            image_id = self.image_id
            self.action = STOP
            if self.verbose:
                print("[LOG] invalid image id caused by random. stopping...")

        self.image_id = image_id


class COCO2014Environment(BaseEnvironment):
    def __init__(
        self,
        entry: dict,
        data_dir: str,
        resize_samples: bool = False,
        reference: bool = False,
        random: bool = False,
        verbose: bool = False,
    ):
        self.direction = entry["direction"]
        self.granularity = entry["granularity"]
        self.horizontal_step = entry["horizontal_step"]
        self.vertical_step = entry["vertical_step"]
        self.crop = entry["crop"]

        max_horizontal_step = 0
        max_vertical_step = 0
        if "zoom" in self.direction:
            hint_prompt = "Hint: zooming out could help with classification."
            action_space = (STOP, ZOOM)
            max_steps = self.granularity + 1
            max_horizontal_step = self.horizontal_step * max_steps
            max_vertical_step = self.vertical_step * max_steps

        else:
            hint_prompt = "Hint: moving the camera could help with classification."

            # if left/right/up/down max steps corresponds to granularity
            # for diagonal movements there are granularity horizontal steps + granularity vertical steps
            max_steps = self.granularity if "_" not in self.direction else self.granularity * 2

            action_space = [STOP]
            if "up" in self.direction:
                action_space.append(UP)
                max_vertical_step = self.vertical_step * self.granularity
            if "down" in self.direction:
                action_space.append(DOWN)
                max_vertical_step = self.vertical_step * self.granularity
            if "left" in self.direction:
                action_space.append(LEFT)
                max_horizontal_step = self.horizontal_step * self.granularity
            if "right" in self.direction:
                action_space.append(RIGHT)
                max_horizontal_step = self.horizontal_step * self.granularity
            action_space = tuple(action_space)

        self.num_horizontal_turns = 0
        self.num_vertical_turns = 0
        self.max_horizontal_step = max_horizontal_step
        self.max_vertical_step = max_vertical_step

        index_correct_options = set(
            [
                i
                for i, opt in enumerate(entry["options"])
                if "cannot" not in opt.lower() and "do not" not in opt.lower()
            ]
        )

        if resize_samples:
            print(
                "[bold yellow][Warning][/] resize is true but is not available on coco. ignoring..."
            )

        super().__init__(
            entry=entry,
            data_dir=data_dir,
            resize_samples=resize_samples,
            reference=reference,
            random=random,
            verbose=verbose,
            _hint_prompt=hint_prompt,
            _index_correct_options=index_correct_options,
            _action_space=action_space,
            _max_steps=max_steps,
        )

    def _get_valid_options(self):
        options = copy.deepcopy(self.options)
        if not self.random:
            if self.direction == "zoom":
                if self.horizontal_step == self.max_horizontal_step:
                    assert self.vertical_step == self.max_vertical_step
                    options.remove("Move farther from the object.")
            else:
                if self.horizontal_step >= self.max_horizontal_step:
                    if "Move the camera to the left." in options:
                        options.remove("Move the camera to the left.")
                    if "Move the camera to the right." in options:
                        options.remove("Move the camera to the right.")
                if self.vertical_step >= self.max_vertical_step:
                    if "Move the camera up." in options:
                        options.remove("Move the camera up.")
                    if "Move the camera down." in options:
                        options.remove("Move the camera down.")
        else:
            if self.verbose:
                print("[LOG] Not removing options in random")
        return options

    def get_open_ended_gen_answers(self):
        options, categories = self._get_correct_options_and_categories()
        categories = ", ".join([cat.replace(".", "") for cat in categories])
        answers = "Move the camera, Move farther from the object, " + categories + "."
        return answers

    def _get_transform(self):
        if self.reference:
            return super()._get_transform()
        else:
            crop = copy.deepcopy(self.crop)
            crop["top"] = (
                self.crop["top"] - self.vertical_step
                if "up" in self.direction or "zoom" in self.direction
                else self.crop["top"]
            )
            crop["left"] = (
                self.crop["left"] - self.horizontal_step
                if "left" in self.direction or "zoom" in self.direction
                else self.crop["left"]
            )
            crop["height"] = (
                self.crop["height"] + self.vertical_step
                if "down" in self.direction or "zoom" in self.direction
                else self.crop["height"]
            )
            crop["width"] = (
                self.crop["width"] + self.horizontal_step
                if "right" in self.direction or "zoom" in self.direction
                else self.crop["width"]
            )
            if self.direction == "zoom":
                crop["height"] += self.crop["top"]
                crop["width"] += self.crop["left"]

            def transform(image):
                return image.crop((crop["left"], crop["top"], crop["width"], crop["height"]))

            return transform

    def _get_relative_image_path(self, image_id):
        return self.image_template

    def _fix_corruptions(self, image_path):
        return super()._fix_corruptions(image_path)

    def _get_correct_options_and_categories(self):
        options = []
        categories = []
        if self.reference:
            categories = self.correct_categories

        if self.random:
            options = self.correct_options
            categories = self.correct_categories

        elif self.direction == "zoom":
            options = copy.deepcopy(self.correct_options)
            categories = copy.deepcopy(self.correct_categories)
            if self.horizontal_step == self.max_horizontal_step:
                assert self.vertical_step == self.max_vertical_step
                options.remove("Move farther from the object.")

        else:
            options = copy.deepcopy(self.correct_options)
            categories = copy.deepcopy(self.correct_categories)

            # we cannot move horizontally anymore
            if self.horizontal_step >= self.max_horizontal_step:
                if "Move the camera to the left." in options:
                    options.remove("Move the camera to the left.")
                if "Move the camera to the right." in options:
                    options.remove("Move the camera to the right.")

            # we cannot move vertically anymore
            if self.vertical_step >= self.max_vertical_step:
                if "Move the camera up." in options:
                    options.remove("Move the camera up.")
                if "Move the camera down." in options:
                    options.remove("Move the camera down.")

        return options, categories

    def _prepare_next_state(self):
        horizontal_step = self.horizontal_step
        vertical_step = self.vertical_step
        if self.action in (LEFT, RIGHT):
            self.num_horizontal_turns += 1
            horizontal_step = (
                horizontal_step / self.num_horizontal_turns * (self.num_horizontal_turns + 1)
            )
        elif self.action in (UP, DOWN):
            self.num_vertical_turns += 1
            vertical_step = vertical_step / self.num_vertical_turns * (self.num_vertical_turns + 1)
        elif self.action == ZOOM:
            self.num_horizontal_turns += 1
            self.num_vertical_turns += 1
            horizontal_step = (
                horizontal_step / self.num_horizontal_turns * (self.num_horizontal_turns + 1)
            )
            vertical_step = vertical_step / self.num_vertical_turns * (self.num_vertical_turns + 1)
        else:
            raise ValueError(f"action {self.action} is not a valid action!")
        self.horizontal_step = horizontal_step
        self.vertical_step = vertical_step
